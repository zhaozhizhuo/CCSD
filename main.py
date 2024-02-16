from train_parser import generate_parser
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import os

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
import shutil
import json
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import time
import datetime
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import BertConfig, BertModel, AdamW
from ZY_bert import *
# from ZY_bert import ClassifiyModule,tokenizer
from data_utils import flat_accuracy,format_time,Data_Loader
from transformers import get_linear_schedule_with_warmup
import sklearn.metrics as metrics
from FreeLB import PGD
from cb_loss import FocalLoss,AsymmetricLossOptimized,AsymmetricLoss
from unbalanced_loss.dice_loss_nlp import MultiDSCLoss



# 设定随机种子值，以确保输出是确定的
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


def run(args):
    print(args.model_path)
    model_config = BertConfig.from_pretrained(args.model_path)
    model = ClassifiyModule(Bertmodel_path=args.model_path, Bertmodel_config=model_config)
    model = model.to(args.device)
    lstm = ClassifiyLSTM(Bertmodel_path=args.model_path, Bertmodel_config=model_config)
    lstm = lstm.to(args.device)

    train_dataloader,test_dataloader,val_dataloader,syndromes_text,id2syndrome_dict, label_feature = Data_Loader(args,tokenizer)

    model = model.to(device)

    gpus = [0]
    # model = nn.parallel.DistributedDataParallel(model, device_ids=gpus, output_device=gpus[0])
    model = nn.DataParallel(model.cuda(), device_ids=gpus, output_device=gpus[0])

    optimizer = AdamW(model.parameters(),
                      lr=1e-5,  # args.learning_rate - default is 5e-5
                      eps=1e-8  # args.adam_epsilon  - default is 1e-8
                      )
    criterion = nn.CrossEntropyLoss().to(args.device)
    total_steps = len(train_dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)

    original_tensor = label_feature.reshape(1, 146, 512, 1024)
    label_fea = original_tensor.repeat(len(gpus), 1, 1, 1)

    # 存储训练和评估的 loss、准确率、训练时长等统计指标,
    training_stats = []

    # 统计整个训练时长
    total_t0 = time.time()
    epochs = args.epochs
    modelCount = 0
    for epoch_i in range(1, epochs + 1):
        # noinspection LanguageDetectionInspection
        with open(f'./main_007/{modelCount}_result.txt', 'w',
                  encoding='utf-8') as f:
            # ========================================
            #               Training
            # ========================================

            print('')
            print('======== Epoch {:} / {:} ========'.format(epoch_i, epochs))
            print('Training...')

            f.write('\n')
            f.write('======== Epoch {:} / {:} ========\n'.format(epoch_i, epochs))
            f.write('Training...\n')
            # 统计单次 epoch 的训练时间
            t0 = time.time()

            # 重置每次 epoch 的训练总 loss
            total_train_loss = 0

            # 将模型设置为训练模式。这里并不是调用训练接口的意思
            # dropout、batchnorm 层在训练和测试模式下的表现是不同的 (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
            model.train()

            # Tracking variables
            total_train_accuracy = 0
            total_train_f1 = 0

            preds = None
            out_label_ids = None
            # 训练集小批量迭代
            for step, batch in tqdm(enumerate(train_dataloader)):

                b_input_ids = batch[0].to(args.device)
                b_input_mask = batch[1].to(args.device)
                b_labels = batch[2].to(args.device)
                true_splitNumbers = batch[3].to(args.device)

                b_input_ids_chief = batch[4].to(args.device)
                b_input_mask_chief = batch[5].to(args.device)
                true_splitNumbers_chief = batch[6].to(args.device)

                # 每次计算梯度前，都需要将梯度清 0，因为 pytorch 的梯度是累加的
                model.zero_grad()
                lstm.zero_grad()

                #症状表现
                chief = lstm(b_input_ids_chief,b_input_mask_chief,true_splitNumbers_chief)
                # print('chief',chief.size())

                # 前向传播
                # 该函数会根据不同的参数，会返回不同的值。

                pred = model(b_input_ids,
                             label_fea=label_fea.data,
                             token_type_ids=None,
                             attention_mask=b_input_mask,
                             return_dict=False,
                             true_splitNumbers=true_splitNumbers,
                             chief=chief
                             )

                # # focaloss
                # db_loss = FocalLoss()
                # db = db_loss(pred, b_labels, reduction_override='mean')

                #dscloss
                # Loss = MultiDSCLoss(alpha=1.0, smooth=1.0, reduction="mean")
                # db = Loss(pred,b_labels)

                # labels = torch.nn.functional.one_hot(b_labels, num_classes=146)
                # efl_loss = AsymmetricLossOptimized()
                # db = efl_loss(pred, labels)

                loss = criterion(pred, b_labels)
                loss = loss
                # 累加 loss
                total_train_loss += loss.item()

                # 反向传播
                loss.backward()


                # 将预测结果和 labels 加载到 cpu 中计算
                pred = pred.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                if preds is None:
                    preds = np.argmax(pred, axis=1).flatten()
                    out_label_ids = label_ids.flatten()
                else:
                    preds = np.append(preds, np.argmax(pred, axis=1).flatten(), axis=0)
                    out_label_ids = np.append(out_label_ids, label_ids.flatten(), axis=0)


                if step % 200 == 0:
                    print('  step {}   loss: {}'.format(step, loss))

                # # 反向传播
                # loss.backward()

                # 梯度裁剪，避免出现梯度爆炸情况
                #         torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1.0)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # 更新参数
                optimizer.step()

                # 更新学习率
                scheduler.step()

            # 平均训练误差
            avg_train_loss = total_train_loss / len(train_dataloader)

            # 单次 epoch 的训练时长
            training_time = format_time(time.time() - t0)

            result = metrics.classification_report(out_label_ids.tolist(), preds.tolist(), output_dict=True, zero_division=1)
            # print(result)

            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Accuracy: {0:.5f}".format(result["accuracy"]))
            print(result["macro avg"])

            f.write("\n")
            f.write("  Average training loss: {0:.2f}\n".format(avg_train_loss))
            f.write("  Accuracy: {0:.5f}\n".format(result["accuracy"]))
            f.write(str(result["macro avg"]))
            # f.write(result["macro avg"])
            # ========================================
            #               Validation
            # ========================================
            # 完成一次 epoch 训练后，就对该模型的性能进行验证

            print('')
            print('Running Validation...')

            f.write('\n')
            f.write('Running Validation...\n')
            t0 = time.time()

            # 设置模型为评估模式
            model.eval()

            # Tracking variables
            total_eval_accuracy = 0
            total_eval_f1 = 0
            total_eval_loss = 0
            nb_eval_steps = 0

            preds = None
            out_label_ids = None
            # Evaluate data for one epoch
            for batch in val_dataloader:
                # 将输入数据加载到 gpu 中
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)
                # label_fea = label_model(syndromes_text, id2syndrome_dict, b_labels)
                true_splitNumbers = batch[3].to(args.device)

                # 症状表现
                b_input_ids_chief = batch[4].to(args.device)
                b_input_mask_chief = batch[5].to(args.device)
                true_splitNumbers_chief = batch[6].to(args.device)
                chief = lstm(b_input_ids_chief, b_input_mask_chief, true_splitNumbers_chief)

                # 评估的时候不需要更新参数、计算梯度
                with torch.no_grad():
                    pred = model(b_input_ids,
                                 label_fea=label_fea.data,
                                 token_type_ids=None,
                                 attention_mask=b_input_mask,
                                 return_dict=False,
                                 true_splitNumbers=true_splitNumbers,
                                 chief=chief
                                 )

                    # focaloss
                    # db_loss = FocalLoss()
                    # db = db_loss(pred, b_labels, reduction_override='mean')
                    # dscloss
                    # Loss = MultiDSCLoss(alpha=1.0, smooth=1.0, reduction="mean")
                    # db = Loss(pred,b_labels)

                    # labels = torch.nn.functional.one_hot(b_labels, num_classes=146)
                    # efl_loss = AsymmetricLossOptimized(gamma_neg=4, gamma_pos=0, clip=0.05,
                    #                                    disable_torch_grad_focal_loss=True)
                    # db = efl_loss(pred, labels)

                    loss = criterion(pred, b_labels)
                    loss = loss

                # 累加 loss
                total_eval_loss += loss.item()

                # 将预测结果和 labels 加载到 cpu 中计算
                pred = pred.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                if preds is None:
                    preds = np.argmax(pred, axis=1).flatten()
                    out_label_ids = label_ids.flatten()
                else:
                    preds = np.append(preds, np.argmax(pred, axis=1).flatten(), axis=0)
                    out_label_ids = np.append(out_label_ids, label_ids.flatten(), axis=0)

            # 统计本次 epoch 的 loss
            avg_val_loss = total_eval_loss / len(val_dataloader)

            # 打印本次 epoch 的准确率
            result = metrics.classification_report(out_label_ids.tolist(), preds.tolist(), output_dict=True, zero_division=1)
            # print(result)
            print("")
            print("  Average eval loss: {0:.2f}".format(avg_val_loss))
            print("  Accuracy: {0:.5f}".format(result["accuracy"]))
            print(result["macro avg"])

            f.write("\n")
            f.write("  Average eval loss: {0:.2f}\n".format(avg_val_loss))
            f.write("  Accuracy: {0:.5f}\n".format(result["accuracy"]))
            f.write(str(result["macro avg"]))

            print('')
            print('Running test...')
            t0 = time.time()

            # 设置模型为评估模式
            model.eval()

            # Tracking variables
            total_eval_accuracy = 0
            total_eval_f1 = 0
            total_eval_loss = 0
            nb_eval_steps = 0

            preds = None
            out_label_ids = None
            # Evaluate data for one epoch
            for batch in test_dataloader:
                # 将输入数据加载到 gpu 中
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)
                # label_fea = label_model(syndromes_text, id2syndrome_dict, b_labels)
                true_splitNumbers = batch[3].to(args.device)

                # 症状表现
                b_input_ids_chief = batch[4].to(args.device)
                b_input_mask_chief = batch[5].to(args.device)
                true_splitNumbers_chief = batch[6].to(args.device)
                chief = lstm(b_input_ids_chief, b_input_mask_chief, true_splitNumbers_chief)

                # 评估的时候不需要更新参数、计算梯度
                with torch.no_grad():
                    pred = model(b_input_ids,
                                 label_fea=label_fea.data,
                                 token_type_ids=None,
                                 attention_mask=b_input_mask,
                                 return_dict=False,
                                 true_splitNumbers=true_splitNumbers,
                                 chief=chief
                                 )
                    # focaloss
                    # db_loss = FocalLoss()
                    # db = db_loss(pred, b_labels, reduction_override='mean')
                    # dscloss
                    # Loss = MultiDSCLoss(alpha=1.0, smooth=1.0, reduction="mean")
                    # db = Loss(pred,b_labels)

                    # labels = torch.nn.functional.one_hot(b_labels, num_classes=146)
                    # efl_loss = AsymmetricLossOptimized(gamma_neg=4, gamma_pos=0, clip=0.05,
                    #                                    disable_torch_grad_focal_loss=True)
                    # db = efl_loss(pred, labels)

                    loss = criterion(pred, b_labels)
                    loss = loss

                # 累加 loss
                total_eval_loss += loss.item()

                # 将预测结果和 labels 加载到 cpu 中计算
                pred = pred.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                if preds is None:
                    preds = np.argmax(pred, axis=1).flatten()
                    out_label_ids = label_ids.flatten()
                else:
                    preds = np.append(preds, np.argmax(pred, axis=1).flatten(), axis=0)
                    out_label_ids = np.append(out_label_ids, label_ids.flatten(), axis=0)

            # 统计本次 epoch 的 loss
            avg_val_loss = total_eval_loss / len(test_dataloader)

            # 打印本次 epoch 的准确率
            result = metrics.classification_report(out_label_ids.tolist(), preds.tolist(), output_dict=True, zero_division=1)
            # print(result)
            print("")
            print("  Average eval loss: {0:.2f}".format(avg_val_loss))
            print("  Accuracy: {0:.5f}".format(result["accuracy"]))
            print(result["macro avg"])

            f.write('\n')
            f.write('Running test...\n')
            f.write("\n")
            f.write("  Average test loss: {0:.2f}\n".format(avg_val_loss))
            f.write("  Accuracy: {0:.5f}\n".format(result["accuracy"]))
            f.write(str(result["macro avg"]))


            # 统计本次评估的时长
            validation_time = format_time(time.time() - t0)

            # f.write("  Validation Loss: {0:.2f}\n".format(avg_val_loss))
            # f.write("  Validation took: {:}\n".format(validation_time))
            # f.write("  Validation Loss: {0:.2f}\n".format(avg_val_loss))
            # f.write("  macro f1: {:}\n".format(total_eval_f1 / len(val_dataloader)))
            # f.write("  Validation took: {:}\n".format(validation_time))
            # f.write("\n")
            # f.write("  Average training loss: {0:.2f}\n".format(avg_train_loss))
            # f.write("  Accuracy: {0:.5f}\n".format(result["accuracy"]))
            # f.write(result["macro avg"])
            # 记录本次 epoch 的所有统计信息
            training_stats.append(
                {
                    'epoch': epoch_i + 1,
                    'Training Loss': avg_train_loss,
                    'Valid. Loss': avg_val_loss,
                    'Valid. Accur.': result["accuracy"],
                    'Training Time': training_time,
                    'Validation Time': validation_time
                }
            )
            torch.save(model,
                       './main_007/model_{}.pkl'.format(
                           modelCount))
            modelCount += 1
    print("")
    print("Training complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

def main():
    parser = generate_parser()
    args = parser.parse_args()
    run(args)

if __name__ == "__main__":
    main()