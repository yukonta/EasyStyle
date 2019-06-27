from neural_style.transformer_net import TransformerNet
from neural_style import utils

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from scipy import misc

import os
import sys
import time
import re
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from neural_style.vgg import Vgg16

# В данном классе мы хотим полностью производить всю обработку картинок, которые поступают к нам из телеграма.

class StyleTransferModel:
    def __init__(self):
        # Сюда необходимо перенести всю иницализацию, вроде загрузки свеерточной сети и т.д.
      
        use_gpu = torch.cuda.is_available()
        if not use_gpu:
            self.arg_cuda = 0
            print('CUDA is not available.  Use CPU ...')
        else:
            self.arg_cuda = 1
            print('CUDA is available!  Use GPU ...')

        self.device = torch.device("cuda" if use_gpu else "cpu")
        print(self.device)

        #imsize = 128
        imsize = 256
        self.loader = transforms.Compose([
            transforms.Resize(imsize),  # нормируем размер изображения
            ##transforms.CenterCrop(imsize),
            #    transforms.ToTensor()
        ])  # превращаем в удобный формат

        pass

    def transfer_style(self, content_img_stream, style_img_stream, style_type):
        # В телеграм боте мы получаем поток байтов BytesIO,
        # а мы хотим спрятать в этот метод всю работу с картинками, поэтому лучше принимать тут эти самые потоки
        # и потом уже приводить их к PIL, а потом и к тензору, который уже можно отдать модели.

        style_img = Image.open(style_img_stream)
        #self.train(dataset = 'neural_style/dataset_dir', style_img = style_img, save_model_dir = 'neural_style/save_model_dir', checkpoint_model_dir = 'neural_style/checkpoint_model_dir', epochs=2, batch_size=4, image_size=256, seed=42, arg_cuda=0, content_weight=1e5, style_weight=1e10, lr=1e-3, log_interval=500, checkpoint_interval=2000)
        if (style_type == 'own'):
            fin_model_dict = self.train(dataset='neural_style/dataset_dir', style_img=style_img,
                                        save_model_dir='neural_style/save_model_dir',
                                        checkpoint_model_dir='neural_style/checkpoint_model_dir',
    #                                    epochs=2, batch_size=8, image_size=64,log_interval=2,checkpoint_interval=2,
                                        epochs=10, batch_size=4, image_size=256, log_interval=50, checkpoint_interval=500,
                                        seed=42, arg_cuda=self.arg_cuda, content_weight=1e5, style_weight=1e10, lr=1e-3
                                        )
        else: #style_type == 'candy' OR 'mosaic' OR 'rain_princess' OR 'udnie'
            model_file_name = 'neural_style/saved_models/' + style_type + '.pth'
            fin_model_dict = torch.load(model_file_name) #NNN

        return misc.toimage(self.process_image(content_img_stream,fin_model_dict)[0])


    def process_image(self, img_stream, fin_model_dict):
        device = self.device
        image = Image.open(img_stream)
        image = self.loader(image)

        out_image = self.stylize(content_img=image, scale=1,
                                 #model_file_name='neural_style/save_model_dir/StyleTransTan.pth',
                                 fin_model_dict = fin_model_dict,
                                 arg_cuda=self.arg_cuda)
        device = torch.device("cpu")
        return out_image.to(device, torch.float)

    def stylize(self, content_img, scale,
               # model_file_name,
                fin_model_dict,
                arg_cuda=0):
        device = torch.device("cuda" if arg_cuda else "cpu")

        # content_image = utils.load_image(args.content_image, scale = content_scale)
        content_image = content_img  # NNN

        if scale is not None:
            content_image = content_image.resize(
                (int(content_image.size[0] / scale), int(content_image.size[1] / scale)), Image.ANTIALIAS)
        content_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])
        content_image = content_transform(content_image)
        content_image = content_image.unsqueeze(0).to(device)
        with torch.no_grad():
            style_model = TransformerNet()

            #state_dict = torch.load(model_file_name) #NNN
            state_dict = fin_model_dict  #NNN


            # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
            for k in list(state_dict.keys()):
                if re.search(r'in\d+\.running_(mean|var)$', k):
                    del state_dict[k]

            style_model.load_state_dict(state_dict)
            style_model.to(device)
            output = style_model(content_image).cpu()
            # utils.save_image('neural_style/output_image_folder/transformed_pict1.jpg', output[0])

        return output
        # utils.save_image(args.output_image, output[0])


    def train(self, dataset, style_img, save_model_dir, checkpoint_model_dir, epochs = 2, batch_size = 4, image_size = 256,
              seed = 42, arg_cuda = 0, content_weight = 1e5, style_weight = 1e10, lr = 1e-3, log_interval = 500, checkpoint_interval = 2000):

        device = torch.device("cuda" if arg_cuda else "cpu")

        np.random.seed(seed)
        torch.manual_seed(seed)

        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])
        train_dataset = datasets.ImageFolder(dataset, transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        print('Train dataset is loaded')
        transformer = TransformerNet().to(device)
        optimizer = Adam(transformer.parameters(), lr)
        mse_loss = torch.nn.MSELoss()
        vgg = Vgg16(requires_grad=False).to(device)
        style_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])
        #style = utils.load_image(args.style_image, size= None) # NNN
        style = style_img # NNN

        style = style_transform(style)
        style = style.repeat(batch_size, 1, 1, 1).to(device)

        features_style = vgg(utils.normalize_batch(style))
        gram_style = [utils.gram_matrix(y) for y in features_style]
        print('Epochs = ',epochs)
        for e in range(epochs):
            transformer.train()
            agg_content_loss = 0.
            agg_style_loss = 0.
            count = 0
            for batch_id, (x, _) in enumerate(train_loader):
                #print('Epoch=', e+1, 'Batch_id=', batch_id)
                n_batch = len(x)
                count += n_batch
                optimizer.zero_grad()

                x = x.to(device)
                y = transformer(x)

                y = utils.normalize_batch(y)
                x = utils.normalize_batch(x)

                features_y = vgg(y)
                features_x = vgg(x)

                content_loss = content_weight * mse_loss(features_y.relu2_2, features_x.relu2_2)

                style_loss = 0.
                for ft_y, gm_s in zip(features_y, gram_style):
                    gm_y = utils.gram_matrix(ft_y)
                    style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])
                style_loss *= style_weight

                total_loss = content_loss + style_loss
                total_loss.backward()
                optimizer.step()

                agg_content_loss += content_loss.item()
                agg_style_loss += style_loss.item()

                if (batch_id + 1) % log_interval == 0:
                    mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                        time.ctime(), e + 1, count, len(train_dataset),
                                      agg_content_loss / (batch_id + 1),
                                      agg_style_loss / (batch_id + 1),
                                      (agg_content_loss + agg_style_loss) / (batch_id + 1)
                    )
                    print(mesg)

                if checkpoint_model_dir is not None and (batch_id + 1) % checkpoint_interval == 0:
                    transformer.eval().cpu()
                    ckpt_model_filename = "ckpt_epoch_" + str(e) + "_batch_id_" + str(batch_id + 1) + ".pth"
                    ckpt_model_path = os.path.join(checkpoint_model_dir, ckpt_model_filename)
                    torch.save(transformer.state_dict(), ckpt_model_path)
                    transformer.to(device).train()

        # save model
        transformer.eval().cpu()
        save_model_filename = "epoch_" + str(epochs) + "_" + str(time.ctime()).replace(' ', '_') + "_" + str(
            content_weight) + "_" + str(style_weight) + ".model"
        save_model_path = os.path.join(save_model_dir, save_model_filename)
        #torch.save(transformer.state_dict(), save_model_path) #NNN
        #torch.save(transformer.state_dict(), 'neural_style/save_model_dir/StyleTransTan.model') #NNN
        fin_model_dict = transformer.state_dict()
        torch.save(fin_model_dict, 'neural_style/save_model_dir/StyleTransTan.pth')  # NNN

        print("\nDone, trained model saved at", save_model_path)
        return fin_model_dict