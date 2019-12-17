import torch
import math

# evaluate MAE (for test or validation phase)
def eval_mae( y_pred, y):
    return torch.abs(y_pred - y).mean()

# TODO: write a more efficient version
# get precisions and recalls: threshold---divided [0, 1] to num values
def prec_recall( y_pred, y_true):

    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)


    return precision,recall

def max_f(average_prec, average_rec):
    beta2 = math.sqrt(0.3)  # for max F_beta metric
    score = (1 + beta2 ** 2) * average_prec * average_rec / (beta2 ** 2 * average_prec + average_rec)
    score[score != score] = 0  # delete the nan
    return score

def eval_pr( y_pred, y, num):
    prec, recall = torch.zeros(num), torch.zeros(num)
    thlist = torch.linspace(0, 1 - 1e-10, num)
    for i in range(num):
        y_temp = (y_pred >= thlist[i]).float()
        tp = (y_temp * y).sum()
        prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / y.sum()
    return prec, recall

# test phase: using origin image size, evaluate MAE and max F_beta metrics
def test(num,output_path, use_crf=False):
    if use_crf: from tools.crf_process import crf
    avg_mae, img_num = 0.0, len(self.test_dataset)
    avg_prec, avg_recall = torch.zeros(num), torch.zeros(num)
    with torch.no_grad():
        counter=0
        for i,data in enumerate(self.test_dataset): #(img, labels) in enumerate(self.test_dataset):
            images,labels = data['image'], data['label']
            images = images.type(torch.cuda. FloatTensor)
            labels= labels.type(torch.cuda.FloatTensor)
            #images = self.transform(img).unsqueeze(0)
            #labels = labels.unsqueeze(0)
            shape = labels.size()[2:]
            #print(shape)
            images = images.to(self.device)
            labels=labels.to(self.device)
            prob_pred = self.net(images)

            prob_pred = torch.mean(torch.cat([prob_pred[i] for i in self.select], dim=1), dim=1, keepdim=True)

            if use_crf:
                prob_pred = crf(img, prob_pred.numpy(), to_tensor=True)
            mae = self.eval_mae(prob_pred, labels)  .item()

            prob_pred = F.interpolate(prob_pred, size=shape, mode='bilinear', align_corners=True)
            ratio = 160/224*7
            #plot_result.append(images[0])
            #plot_result.append(labels[0])
            result_dir=output_path
            #plot_image(prob_pred[0], (224/60, 224/60), 'Predicted Map')
            print(images.size()[0])
            for j in range(images.size()[0]):
                print(counter)
            #    plot_image(images[j], (224/120, 224/120), 'Input Image',True)
            #    plot_image(labels[j], (224/120, 224/120), 'Ground Truth')
            #    plot_image(prob_pred[j], (224/120, 224/120), 'Predicted Map')
                save_image(images[j],result_dir+'input'+str(counter)+'.jpg')
                save_image(make_grid([labels[j],prob_pred[j]]),result_dir+'result'+str(counter)+'.png')
                counter = counter +1
            prec, recall = self.prec_recall(prob_pred, labels, num)

            #print(num)
            print("[%d] mae: %.4f" % (i, mae))
            print("[%d] mae: %.4f" % (i, mae), file=self.test_output)
            avg_mae += mae
            avg_prec, avg_recall = avg_prec + prec, avg_recall + recall

    avg_mae, avg_prec, avg_recall = avg_mae / img_num, avg_prec / img_num, avg_recall / img_num

    score = (1 + self.beta ** 2) * avg_prec * avg_recall / (self.beta ** 2 * avg_prec + avg_recall)
    score[score != score] = 0  # delete the nan
    print('average mae: %.4f, max fmeasure: %.4f' % (avg_mae, score.max()))
    print('average mae: %.4f, max fmeasure: %.4f' % (avg_mae, score.max()), file=self.test_output)
