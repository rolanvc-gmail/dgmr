import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import os
from norms import Norm_1_torch
from spatialdiscriminator import spaDiscriminator
from temporaldiscriminator import temDiscriminator
from generator import Generator
import glob as glob
import matplotlib.pyplot as plt
import matplotlib
import random
import sys
cuda = True if torch.cuda.is_available() else False
os.environ["CUDA_VISIBLE_DEVICES"] = '1,2'
device=1
torch.cuda.device(device)

BATCHSIZE = 16
REP = 4
M = 4
N = 22
H = 256
W = 256
Lambda = 20
num_epoch = 5
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
RELU = nn.ReLU()

vmin = -30
vmax = 75

norm = plt.Normalize(vmin, vmax)
cmap = matplotlib.cm.get_cmap('jet')
sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

train_mos = ['01', '03', '05', '07', '08', '09']
test_mos = ['02', '04', '06']


def create_dummy_real_sequence():
    data = []
    for i in range(BATCHSIZE):
        img_set = []
        mos = random.choice(train_mos)
        root = '/home/rolan/Weather-Datasets/npy-data/' + str(mos) + '/'
        dirs = [item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item))]
        day = random.choice(dirs)
        d = '/home/rolan/Weather-Datasets/npy-data/' + str(mos) + '/' + str(day) + '/'
        files = glob.glob(d + '*.npy')
        start = np.random.randint(0, len(files) - 22)
        for j in range(22):
            im = np.load(files[start + j])
            im = im[0]
            pic = np.zeros((256, 256)) + im
            pic = np.expand_dims(pic, axis=2)
            img_set.append(pic)
        data.append(img_set)       
    return np.array(data)[:, :, :, :, 0].astype(np.float32)


def create_dummy_real_sequence_for_gen():
    data = []
    for i in range(int(BATCHSIZE / REP)):
        img_set = []
        mos = random.choice(train_mos)
        root = '/home/rolan/Weather-Datasets/npy-data/' + str(mos) + '/'
        dirs = [item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item))]
        day = random.choice(dirs)
        d = '/home/rolan/Weather-Datasets/npy-data/' + str(mos) + '/' + str(day) + '/'
        files = glob.glob(d + '*.npy')
        start = np.random.randint(0, len(files) - 22)
        for j in range(22):
            im = np.load(files[start + j])
            im = im[0]
            pic = np.zeros((256, 256)) + im
            pic = np.expand_dims(pic, axis=2)
            img_set.append(pic)
        for k in range(REP):
            data.append(img_set)       
    return np.array(data)[:, :, :, :, 0].astype(np.float32)


###############################################################################################
###############################################################################################
###############################################################################################
if __name__ == "__main__":

    sd = spaDiscriminator()
    td = temDiscriminator()
    g = Generator(24)
    sig = nn.Sigmoid()
    
    if torch.cuda.is_available():
        sd = sd.cuda(device)
        td = td.cuda(device)
        g = g.cuda(device)

    if not os.path.exists('./model'):
        os.makedirs('./model')

    if not os.path.exists('./actual'):
        os.makedirs('./actual')

    if not os.path.exists('./fake'):
        os.makedirs('./fake')

    if os.path.exists('./model/sd_.dict') and os.path.exists('./model/td_.dict') and os.path.exists('./model/g_.dict'):
        print("loading saved model")
        sd.load_state_dict(torch.load('./model/sd_.dict'))
        td.load_state_dict(torch.load('./model/td_.dict'))
        g.load_state_dict(torch.load('./model/g_.dict'))
        print("saved model loaded")
        print("")
    
    sd_optimizer = torch.optim.Adam(sd.parameters(), betas=(0.0, 0.999), lr=0.0002)
    td_optimizer = torch.optim.Adam(td.parameters(), betas=(0.0, 0.999), lr=0.0002)
    g_optimizer = torch.optim.Adam(g.parameters(), betas=(0.0, 0.999), lr=0.00005)
    # sd_optimizer = torch.optim.RMSprop(sd.parameters(), lr=0.000001, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
    # td_optimizer = torch.optim.RMSprop(td.parameters(), lr=0.000001, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
    # g_optimizer = torch.optim.RMSprop(g.parameters(), lr=0.000001, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
    real_label = Variable(torch.ones(BATCHSIZE)).cuda(device)
    fake_label = Variable(torch.zeros(BATCHSIZE)).cuda(device)

    filename = "./counter.txt"
    if os.path.isfile(filename):
        iter_file = open(filename, "rb")
        start = int.from_bytes(iter_file.read(), sys.byteorder)
    else:
        start = 0

    for e in range(start, 5000000):
        iter_file = open("counter.txt", "wb", 0)
        iter_file.write(e.to_bytes(4, byteorder=sys.byteorder))
        print("iteration " + str(e))
        
        for j in range(2):
            S = random.sample(range(0, 18), 8)  # spatial discriminator picks uniformly at random 8 out of 18 lead times

            # train discriminators on real inputs
            data = create_dummy_real_sequence()  # FETCH DATA HERE
            input_real_1sthalf = Variable(torch.from_numpy(data[:, :4])).cuda(device=device)  # 2 x 4 x 256 x 256 x 1 (for generator)
            input_real_2ndhalf = Variable(torch.from_numpy(data[:, 4:])).cuda(device=device)  # 2 x 18 x 256 x 256 x 1 (for spatial discriminator)
            input_real_whole = Variable(torch.from_numpy(data)).cuda(device=device)  # 2 x 22 x 256 x 256 x 1 (for temporal discriminator)
            input_real_2ndhalf_sd = input_real_2ndhalf[:, S]
            sd_pred_real = sd(input_real_2ndhalf_sd)  # output of spatial discriminator for real lead times x 18
            td_pred_real = td(input_real_whole)  # output of temporal discriminator for entire sequence x 22

            # train discriminators on fake inputs
            z = Variable(Tensor(np.random.normal(0, 1, (BATCHSIZE, 8, 8, 8)))).cuda(device=device)  # latent variable input for latent conditioning stack
            fake_img = g(input_real_1sthalf, z).detach()  # fake output of generator x 18
            fake_img_2ndhalf_sd = fake_img[:, S]  # get input to spatial discriminator from fake images
            sd_pred_fake = sd(fake_img_2ndhalf_sd)
            fake_img_whole_td = torch.cat((input_real_1sthalf, fake_img), dim=1)  # create input to temporal discriminator from fake images
            td_pred_fake = td(fake_img_whole_td)
            
            sd_loss = torch.mean(RELU(1 - sd_pred_real) + RELU(1 + sd_pred_fake))
            td_loss = torch.mean(RELU(1 - td_pred_real) + RELU(1 + td_pred_fake))
            # sd_loss_real = torch.log(sig(sd_pred_real) + 1e-5)
            # td_loss_real = torch.log(sig(td_pred_real) + 1e-5)
            # sd_loss_fake = torch.log(1-sig(sd_pred_fake) + 1e-5)
            # td_loss_fake = torch.log(1-sig(td_pred_fake) + 1e-5)
            # sd_loss = -torch.mean(sd_loss_real + sd_loss_fake)
            # td_loss = -torch.mean(td_loss_real + td_loss_fake)
            # sd_loss = torch.mean(sig(sd_pred_real)) - torch.mean(sig(sd_pred_fake))
            # td_loss = torch.mean(sig(td_pred_real)) - torch.mean(sig(td_pred_fake))
            
            d_loss = (sd_loss + td_loss) 
            sd_optimizer.zero_grad() 
            td_optimizer.zero_grad() 
            d_loss.backward() 
            sd_optimizer.step()
            td_optimizer.step()
            
            # with torch.no_grad():
            #    for v in sd.parameters():
            #        v[:] = v.clip(-0.01, +0.01)
            #    for v in td.parameters():
            #        v[:] = v.clip(-0.01, +0.01)
                
        print("discriminator loss: " + str(np.mean(d_loss.detach().cpu().numpy())))
        
        # train generator
        data = create_dummy_real_sequence_for_gen()
        
        input_real_1sthalf = Variable(torch.from_numpy(data[:, :4])).cuda(device)  # 2 x 4 x 256 x 256 x 1 (for generator)
        input_real_2ndhalf = Variable(torch.from_numpy(data[:, 4:])).cuda(device)  # 2 x 18 x 256 x 256 x 1 (for generator true values)
        z = Variable(Tensor(np.random.normal(0, 1, (BATCHSIZE, 8, 8, 8)))).cuda(device)
        g_output = g(input_real_1sthalf, z)  # input is real stack of 4 images
 
        fake_img_2ndhalf_sd_g = g_output[:, S]
        sd_pred_fake_g = sd(fake_img_2ndhalf_sd_g)  # get prediction of spatial discriminator
        
        fake_img_whole_td_g = torch.cat((input_real_1sthalf, g_output), dim=1)
        td_pred_fake_g = td(fake_img_whole_td_g)  # get prediction of temporal discriminator
        
        r_loss_sum = 0  # compute r loss for generator
        for i in range(BATCHSIZE):
            result = torch.mul((g_output[i] - input_real_2ndhalf[i]), input_real_2ndhalf)
            r_loss = (1 / H * W * N) * Lambda * Norm_1_torch(result)
            r_loss_sum = r_loss_sum + r_loss

        # g_loss_sum = - (torch.mean(sig(sd_pred_fake_g) + sig(td_pred_fake_g)))
        # g_loss_sum = - (torch.mean(sig(sd_pred_fake_g) + sig(td_pred_fake_g)))
        g_loss_sum = -(torch.mean(sd_pred_fake_g) + torch.mean(td_pred_fake_g)) + (r_loss_sum / BATCHSIZE)
        
        if e % 25 == 0:
            pred = g_output.detach().cpu().numpy()[0]
            
            for i in range(data[0].shape[0]):
                fig = plt.figure()
                ax = fig.add_subplot(111)
                cbar = fig.colorbar(sm)
                cbar.ax.set_title("dbz")
                ax.imshow(data[0][i], cmap='jet', vmin=vmin, vmax=vmax)
                plt.savefig('actual/actual_' + str(i) + '.png')
            
            for i in range(pred.shape[0]):
                gg = pred[i]
                # gg = gg.reshape(256,256,1) * 255
                # gg = gg.astype(np.uint8)
                # cv2.imwrite(str(i)+'_output.jpg',gg)
                fig = plt.figure()
                ax = fig.add_subplot(111)
                cbar = fig.colorbar(sm)
                cbar.ax.set_title("dbz")
                ax.imshow(gg, cmap='jet', vmin=vmin, vmax=vmax)
                plt.savefig('fake/fake_' + str(i) + '.png')

        g_optimizer.zero_grad()
        g_loss_sum.backward()
        g_optimizer.step()
        
        if e % 1000 == 0:
            print("saving model for iteration: " + str(e))
            torch.save(sd.state_dict(), './model/sd_.dict')
            torch.save(td.state_dict(), './model/td_.dict')
            torch.save(g.state_dict(), './model/g_.dict')
            print("model saved for iteration: " + str(e))
            print("")
        
        # with torch.no_grad():
        #    for v in g.parameters():
        #        v[:] = v.clamp(-0.01, +0.01)
        
        print("generator loss: " + str(np.mean(g_loss_sum.detach().cpu().numpy())))
        pred_sd = sig(sd_pred_fake)
        pred_td = sig(td_pred_fake)
        
        print("sd sample prediction: " + str(pred_sd.detach().cpu().numpy()))
        print("td sample prediction: " + str(pred_td.detach().cpu().numpy()))
        print("")
