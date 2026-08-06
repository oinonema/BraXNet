#!/usr/bin/env python3
""" ImageNet Validation Script

This is intended to be a lean and easily modifiable ImageNet validation script for evaluating pretrained
models or training checkpoints against ImageNet or similarly organized image datasets. It prioritizes
canonical PyTorch, standard Python style, and good performance. Repurpose as you see fit.

Hacked together by Ross Wightman (https://github.com/rwightman)
"""
import os
import csv
import glob
import time
import torch
import logging
import argparse
import numpy as np
from tqdm import tqdm
from torch import nn
from contextlib import suppress
from collections import OrderedDict
from sklearn.metrics import (confusion_matrix, classification_report, 
                             precision_score, recall_score, f1_score,
                             roc_auc_score)

from timm.models import create_model, apply_test_time_pool, load_checkpoint, is_model, list_models
from timm.data import create_dataset, create_loader, resolve_data_config, RealLabelsImagenet
from timm.utils import accuracy, AverageMeter, natural_key, setup_default_logging, set_jit_legacy

import models

has_apex = False
try:
    from apex import amp
    has_apex = True
except ImportError:
    pass

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('validate')


parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('--data', metavar='DIR', default='/clas/data/class2',#
                    help='path to dataset')
parser.add_argument('--dataset', '-d', metavar='NAME', default='',
                    help='dataset type (default: ImageFolder/ImageTar if empty)')
parser.add_argument('--split', metavar='NAME', default='test',
                    help='dataset split (default: test)')
parser.add_argument('--dataset-download', action='store_true', default=False,
                    help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
parser.add_argument('--model', '-m', metavar='NAME', default='BraXNet_t',#
                    help='model architecture')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--img-size', default=224, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop pct')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float,  nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--num-classes', type=int, default=4,
                    help='Number classes in dataset')
parser.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                    help='path to class to idx mapping file (default: "")')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--log-freq', default=25, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--checkpoint', default='..../model_best.pth.tar', type=str, metavar='PATH', #
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--test-pool', dest='test_pool', action='store_true',
                    help='enable test time pool')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--pin-mem', action='store_true', default=True,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--amp', action='store_true', default=False,
                    help='Use AMP mixed precision. Defaults to Apex, fallback to native Torch AMP.')
parser.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--native-amp', action='store_true', default=False,
                    help='Use Native Torch AMP mixed precision')
parser.add_argument('--tf-preprocessing', action='store_true', default=False,
                    help='Use Tensorflow preprocessing pipeline (require CPU TF installed')
parser.add_argument('--use-ema', dest='use_ema', action='store_true',
                    help='use ema version of weights if present')
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')
parser.add_argument('--legacy-jit', dest='legacy_jit', action='store_true',
                    help='use legacy jit mode for pytorch 1.5/1.5.1/1.6 to get back fusion performance')
parser.add_argument('--results-file', default=None, type=str, metavar='FILENAME',
                    help='Output csv file for validation results (summary)')
parser.add_argument('--real-labels', default='', type=str, metavar='FILENAME',
                    help='Real labels JSON file for imagenet evaluation')
parser.add_argument('--valid-labels', default='', type=str, metavar='FILENAME',
                    help='Valid label indices txt file for validation of partial label space')
parser.add_argument('--class-names', default='glioma_tumor,meningioma_tumor,no_tumor,pituitary_tumor', type=str,
                    help='Comma-separated class names for display')


def validate(args):
    # might as well try to validate something
    args.pretrained = args.pretrained or not args.checkpoint
    args.prefetcher = not args.no_prefetcher
    amp_autocast = suppress  # do nothing
    if args.amp:
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
        else:
            _logger.warning("Neither APEX or Native Torch AMP is available.")
    assert not args.apex_amp or not args.native_amp, "Only one AMP mode should be set."
    if args.native_amp:
        amp_autocast = torch.cuda.amp.autocast
        _logger.info('Validating in mixed precision with native PyTorch AMP.')
    elif args.apex_amp:
        _logger.info('Validating in mixed precision with NVIDIA APEX AMP.')
    else:
        _logger.info('Validating in float32. AMP not enabled.')

    if args.legacy_jit:
        set_jit_legacy()

    # create model
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        in_chans=3,
        global_pool=args.gp,
        scriptable=args.torchscript)
    
    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, args.use_ema, strict=False)

    param_count = sum([m.numel() for m in model.parameters()])
    _logger.info('Model %s created, param count: %d' % (args.model, param_count))

    data_config = resolve_data_config(vars(args), model=model, use_test_size=True, verbose=True)
    test_time_pool = False
    if args.test_pool:
        model, test_time_pool = apply_test_time_pool(model, data_config, use_test_size=True)

    if args.torchscript:
        torch.jit.optimized_execution(True)
        model = torch.jit.script(model)

    model = model.cuda()
    if args.apex_amp:
        model = amp.initialize(model, opt_level='O1')

    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu)))

    criterion = nn.CrossEntropyLoss().cuda()

    dataset = create_dataset(
        root=args.data, name=args.dataset, split=args.split,
        download=args.dataset_download, load_bytes=args.tf_preprocessing, class_map=args.class_map)

    if args.valid_labels:
        with open(args.valid_labels, 'r') as f:
            valid_labels = {int(line.rstrip()) for line in f}
            valid_labels = [i in valid_labels for i in range(args.num_classes)]
    else:
        valid_labels = None

    if args.real_labels:
        real_labels = RealLabelsImagenet(dataset.filenames(basename=True), real_json=args.real_labels)
    else:
        real_labels = None

    crop_pct = 1.0 if test_time_pool else data_config['crop_pct']
    loader = create_loader(
        dataset,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        crop_pct=crop_pct,
        pin_memory=args.pin_mem,
        tf_preprocessing=args.tf_preprocessing)

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    # 收集所有预测和标签用于计算医学指标
    all_preds = []
    all_targets = []
    all_probs = []

    model.eval()
    with torch.no_grad():
        # warmup, reduce variability of first batch time, especially for comparing torchscript vs non
        input = torch.randn((args.batch_size,) + tuple(data_config['input_size'])).cuda()
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)
        model(input)
        pbar = tqdm(total=len(dataset), desc='Validating')
        end = time.time()
        for input, target in loader:
            if args.no_prefetcher:
                target = target.cuda(non_blocking=args.pin_mem)
                input = input.cuda(non_blocking=args.pin_mem)
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            # compute output
            with amp_autocast():
                output = model(input)
                if valid_labels is not None:
                    output = output[:, valid_labels]
                loss = criterion(output, target)
            if real_labels is not None:
                real_labels.add_result(output)

            # 收集预测结果用于医学指标
            probs = torch.softmax(output, dim=1)
            preds = output.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output.detach(), target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1.item(), input.size(0))
            top5.update(acc5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            pbar.update(args.batch_size)
            
    pbar.close()
    if real_labels is not None:
        # real labels mode replaces topk values at the end
        top1a, top5a = real_labels.get_accuracy(k=1), real_labels.get_accuracy(k=5)
    else:
        top1a, top5a = top1.avg, top5.avg

    # ==================== 计算医学分类指标 ====================
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    # Precision, Recall, F1 (macro平均)
    precision_macro = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall_macro = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1_macro = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    
    # Weighted平均 (考虑类别不平衡)
    precision_weighted = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    recall_weighted = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    f1_weighted = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    
    # 计算AUC (多分类使用ovr策略)
    try:
        auc_macro = roc_auc_score(all_targets, all_probs, multi_class='ovr', average='macro')
        auc_weighted = roc_auc_score(all_targets, all_probs, multi_class='ovr', average='weighted')
    except:
        auc_macro = 0.0
        auc_weighted = 0.0
    
    # 混淆矩阵
    cm = confusion_matrix(all_targets, all_preds)
    
    # 计算每个类别的指标
    precision_per_class = precision_score(all_targets, all_preds, average=None, zero_division=0)
    recall_per_class = recall_score(all_targets, all_preds, average=None, zero_division=0)
    f1_per_class = f1_score(all_targets, all_preds, average=None, zero_division=0)
    
    # 计算特异性 (Specificity) - 每个类别
    specificity_per_class = []
    for i in range(args.num_classes):
        tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
        fp = np.sum(cm[:, i]) - cm[i, i]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificity_per_class.append(specificity)
    specificity_macro = np.mean(specificity_per_class)
    
    # 打印详细的医学分类报告
    _logger.info('医学分类评估报告')
    _logger.info(f'\n基本指标:')
    _logger.info(f'  Accuracy: {top1a:.2f}%')
    _logger.info(f'  Loss: {losses.avg:.4f}')
    
    _logger.info(f'\n宏平均指标 (Macro Average):')
    _logger.info(f'  Precision (精确率): {precision_macro*100:.2f}%')
    _logger.info(f'  Recall/Sensitivity (召回率/敏感度): {recall_macro*100:.2f}%')
    _logger.info(f'  Specificity (特异性): {specificity_macro*100:.2f}%')
    _logger.info(f'  F1-Score: {f1_macro*100:.2f}%')
    _logger.info(f'  AUC-ROC: {auc_macro:.4f}')
    
    _logger.info(f'\n加权平均指标 (Weighted Average):')
    _logger.info(f'  Precision: {precision_weighted*100:.2f}%')
    _logger.info(f'  Recall: {recall_weighted*100:.2f}%')
    _logger.info(f'  F1-Score: {f1_weighted*100:.2f}%')
    _logger.info(f'  AUC-ROC: {auc_weighted:.4f}')
    
    # 获取类别名称
    class_names = args.class_names.split(',') if args.class_names else [str(i) for i in range(args.num_classes)]
    
    _logger.info(f'\n各类别详细指标:')
    _logger.info(f'  {"Class":<25} {"Precision":<12} {"Recall":<12} {"Specificity":<12} {"F1-Score":<12}')
    _logger.info(f'  {"-"*73}')
    for i in range(args.num_classes):
        _logger.info(f'  {class_names[i]:<25} {precision_per_class[i]*100:<12.2f} {recall_per_class[i]*100:<12.2f} '
                    f'{specificity_per_class[i]*100:<12.2f} {f1_per_class[i]*100:<12.2f}')
    
    _logger.info(f'\n混淆矩阵 (Confusion Matrix):')
    _logger.info(f'  行=真实标签, 列=预测标签')
    _logger.info(f'  类别: {class_names}')
    _logger.info(f'{cm}')
    
    _logger.info(f'\n详细分类报告:\n')
    _logger.info(classification_report(all_targets, all_preds, target_names=class_names, digits=4))
    _logger.info('='*70 + '\n')
    
    # ==================== 保存评估报告到文件 ====================
    if args.checkpoint:
        output_dir = os.path.dirname(args.checkpoint)
        report_path = os.path.join(output_dir, 'evaluation_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('====医学分类评估报告====\n')
            f.write(f'\n模型: {args.model}\n')
            f.write(f'Checkpoint: {args.checkpoint}\n')
            f.write(f'数据集: {args.data}\n')
            f.write(f'类别数: {args.num_classes}\n')
            f.write(f'样本数: {len(all_targets)}\n')
            
            f.write(f'\n基本指标:\n')
            f.write(f'  Accuracy: {top1a:.2f}%\n')
            f.write(f'  Loss: {losses.avg:.4f}\n')
            
            f.write(f'\n宏平均指标 (Macro Average):\n')
            f.write(f'  Precision (精确率): {precision_macro*100:.2f}%\n')
            f.write(f'  Recall/Sensitivity (召回率/敏感度): {recall_macro*100:.2f}%\n')
            f.write(f'  Specificity (特异性): {specificity_macro*100:.2f}%\n')
            f.write(f'  F1-Score: {f1_macro*100:.2f}%\n')
            f.write(f'  AUC-ROC: {auc_macro:.4f}\n')
            
            f.write(f'\n加权平均指标 (Weighted Average):\n')
            f.write(f'  Precision: {precision_weighted*100:.2f}%\n')
            f.write(f'  Recall: {recall_weighted*100:.2f}%\n')
            f.write(f'  F1-Score: {f1_weighted*100:.2f}%\n')
            f.write(f'  AUC-ROC: {auc_weighted:.4f}\n')
            
            f.write(f'\n======各类别详细指标======:\n')
            f.write(f'  {"Class":<25} {"Precision":<12} {"Recall":<12} {"Specificity":<12} {"F1-Score":<12}\n')
            f.write(f'  {"-"*73}\n')
            for i in range(args.num_classes):
                f.write(f'  {class_names[i]:<25} {precision_per_class[i]*100:<12.2f} {recall_per_class[i]*100:<12.2f} '
                       f'{specificity_per_class[i]*100:<12.2f} {f1_per_class[i]*100:<12.2f}\n')
            
            f.write(f'\n混淆矩阵 (Confusion Matrix):\n')
            f.write(f'  行=真实标签, 列=预测标签\n')
            f.write(f'  类别: {class_names}\n')
            f.write(f'{cm}\n')
            
            f.write(f'\n详细分类报告:\n')
            f.write(classification_report(all_targets, all_preds, digits=4))
            f.write('\n' + '='*70 + '\n')
        
        _logger.info(f'评估报告已保存到: {report_path}')

    results = OrderedDict(
        top1=round(top1a, 4), 
        top1_err=round(100 - top1a, 4),
        top5=round(top5a, 4), 
        top5_err=round(100 - top5a, 4),
        precision_macro=round(precision_macro * 100, 4),
        recall_macro=round(recall_macro * 100, 4),
        specificity_macro=round(specificity_macro * 100, 4),
        f1_macro=round(f1_macro * 100, 4),
        auc_macro=round(auc_macro, 4),
        precision_weighted=round(precision_weighted * 100, 4),
        recall_weighted=round(recall_weighted * 100, 4),
        f1_weighted=round(f1_weighted * 100, 4),
        auc_weighted=round(auc_weighted, 4),
        param_count=round(param_count / 1e6, 2),
        img_size=data_config['input_size'][-1],
        crop_pct=crop_pct,
        interpolation=data_config['interpolation']
    )

    _logger.info(f"Accuracy@top-1 of the network on the {len(dataset)} test images: {top1a:.1f}%")
    return results

def main():
    setup_default_logging()
    args = parser.parse_args()
    model_cfgs = []
    model_names = []
    if os.path.isdir(args.checkpoint):
        # validate all checkpoints in a path with same model
        checkpoints = glob.glob(args.checkpoint + '/*.pth.tar')
        checkpoints += glob.glob(args.checkpoint + '/*.pth')
        model_names = list_models(args.model)
        model_cfgs = [(args.model, c) for c in sorted(checkpoints, key=natural_key)]
    else:
        if args.model == 'all':
            # validate all models in a list of names with pretrained checkpoints
            args.pretrained = True
            model_names = list_models(pretrained=True, exclude_filters=['*_in21k', '*_in22k'])
            model_cfgs = [(n, '') for n in model_names]
        elif not is_model(args.model):
            # model name doesn't exist, try as wildcard filter
            model_names = list_models(args.model)
            model_cfgs = [(n, '') for n in model_names]

        if not model_cfgs and os.path.isfile(args.model):
            with open(args.model) as f:
                model_names = [line.rstrip() for line in f]
            model_cfgs = [(n, None) for n in model_names if n]

    if len(model_cfgs):
        results_file = args.results_file or './results-all.csv'
        _logger.info('Running bulk validation on these pretrained models: {}'.format(', '.join(model_names)))
        results = []
        try:
            start_batch_size = args.batch_size
            for m, c in model_cfgs:
                batch_size = start_batch_size
                args.model = m
                args.checkpoint = c
                result = OrderedDict(model=args.model)
                r = {}
                while not r and batch_size >= args.num_gpu:
                    torch.cuda.empty_cache()
                    try:
                        args.batch_size = batch_size
                        print('Validating with batch size: %d' % args.batch_size)
                        r = validate(args)
                    except RuntimeError as e:
                        if batch_size <= args.num_gpu:
                            print("Validation failed with no ability to reduce batch size. Exiting.")
                            raise e
                        batch_size = max(batch_size // 2, args.num_gpu)
                        print("Validation failed, reducing batch size by 50%")
                result.update(r)
                if args.checkpoint:
                    result['checkpoint'] = args.checkpoint
                results.append(result)
        except KeyboardInterrupt as e:
            pass
        results = sorted(results, key=lambda x: x['top1'], reverse=True)
        if len(results):
            write_results(results_file, results)
    else:
        validate(args)

def write_results(results_file, results):
    with open(results_file, mode='w') as cf:
        dw = csv.DictWriter(cf, fieldnames=results[0].keys())
        dw.writeheader()
        for r in results:
            dw.writerow(r)
        cf.flush()

if __name__ == '__main__':
    main()