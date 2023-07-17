
import torch
from torch import nn
from torchvision import transforms, models
import SignalProc
import os
import numpy as np

def classify(file,sp,imgWidth,device,model):
    # Load the bmp
    sp.readBmp(file,repeat=False,rotate=False,silent=True)

    # Split into pieces
    useClicks = False
    if useClicks:
        res = ClickSearch(sp.sg,sp.sampleRate)
        if res is not None:
            starts = range(res[0], res[1]-imgWidth, imgWidth//2)
    else:
        starts = range(0, np.shape(sp.sg)[1] - imgWidth, imgWidth//2)

    trans = transforms.Compose([
            transforms.Resize(224),
            #transforms.CenterCrop(224),
            #transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    i = 0
    inputs = torch.ones([len(starts),3,784,224],device=torch.device('cpu'))
    for s in starts:
        image = sp.sg[:,s:s+imgWidth].T
        image = torch.from_numpy(image).float()
        # Change 2D into 3D, all channels the same
        image = image.unsqueeze((0)).repeat(3,1,1)
        image = trans(image)
        inputs[i,:,:,:] = image
        i+=1

    # Classify the pieces
    with torch.no_grad():
        inputs = inputs.to(device)
        outputs = model(inputs)
        preds = torch.softmax(outputs, 1)

    return preds[:,:3]

def predict(class_names,model,thr,device,dirname,sp,imgWidth,outf):
    for root, dirs, files in os.walk(dirname):
        for filename in files:
            if filename.endswith('.bmp'):
                print(os.path.join(root,filename))
                y = classify(os.path.join(root,filename),sp,imgWidth,device,model)
                means = np.zeros(len(class_names))
                for c in range(len(class_names)):
                    means[c] = np.mean(np.partition(y[:,c],-4)[-4:])
                inds = np.where(means>thr)
                #outf.write(filename)
                # Convention is that last class is noise
                classes = ""
                for i in range(len(class_names)-1):
                    if i in inds[0]:
                        classes += class_names[i] + " "
                if len(classes)==0:
                    classes = class_names[-1]
                
                # Bit ugly...
                date = filename[:8]
                d1 = date[6:] + '/' + date[4:6] + '/' + date[:4] + ','
                time = filename[9:15]
                t1 = time[:2] + ':' + time[2:4] + ':' + time[5:7] + ','
                if classes == class_names[2]:
                    outf.write(d1+t1+','+classes+','+ root[len(dirname):] + ',' + filename + ',' '\n')
                else:
                    outf.write(d1+t1+','+classes+','+ root[len(dirname):] + ',' + filename + ',AviaNZ' '\n')
                #outf.write(np.array_str(means)+classes+'\n')

#5/11/2019,10:31:23 p.m.,,Unassigned,.\20191105,.\20191105_223123.bmp,

def run():
    class_names = ['Long tail','Short tail','Unassigned']
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load('../bat-res-full224.pth',map_location=torch.device('cpu')))
    #model = models.vgg16(pretrained=True)
    #model.classifier[6].out_features = 3
    #model.load_state_dict(torch.load('bat-vgg128.pth',map_location=torch.device('cpu')))
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    sp = SignalProc.SignalProc(1024,512)
    imgWidth = 224
    
    dirname = '/home/marslast/Dropbox/Transfer/AviaNZ/BatTraining/check/' #Testing/R1/Bat/20191110/'
    outf = open('check.txt','w')
    outf.write('Date,Time,AssignedSite,Category,Foldername,Filename,Observer\n')

    thr = 0.5
    predict(class_names,model,thr,device,dirname,sp,imgWidth,outf)

    outf.close()

run()
