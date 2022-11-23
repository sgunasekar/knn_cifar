import os
import time
from ann import *
import argparse
import torchvision
import torchvision.transforms as transforms
from config import *
from timm.data import rand_augment_transform, Mixup

parser = argparse.ArgumentParser(description='ANN CIFAR10')
parser_add_arguments(parser)


def main():
    args = parser.parse_args()

    device = torch.device(args.device)
    using_gpu = device.type =='cuda'

    root = os.environ.get('DATA_DIR',default_cfg['root'])
    save_dir = os.environ.get('OUTPUT_DIR', default_cfg['save_dir'])

    if using_gpu:
        save_dir = os.path.join(save_dir,"gpu")

    if not os.path.exists(save_dir):
        print("Save directory %s does not exist: creating the directory" %(save_dir))
        os.makedirs(save_dir)

    d = default_cfg['d']

    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mu,sigma)])
    testset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=test_transform)

    testloader = torch.utils.data.DataLoader(testset, shuffle = False, batch_size = len(testset), pin_memory = using_gpu, drop_last = False)

    Xtest,ytest = iter(testloader).next()
    Xtest = Xtest.numpy().reshape(-1,d)
    ytest = ytest.numpy()

    print("Test datasize ", Xtest.shape,ytest.shape)

    if args.use_mixup:
        mixup_fn = Mixup(num_classes=default_cfg['num_classes'], **default_cfg['Mixup_kwargs'])
    else:
        mixup_fn = None

    print("Args: ", args)
    print("Mixup fn: ", mixup_fn)

    for index in args.indexes:
        print("===========================")
        print("Index ", index)

        index_args = default_cfg['index_args_map'][index]
        index_obj = default_cfg['index_map'][index]
        index_suffix = default_cfg['index_suffix_map'][index]

        ann = ANN()

        t = time.time()
        # No augmentation
        transform =  transforms.Compose([transforms.ToTensor(), transforms.Normalize(mu,sigma)])
        trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=False, transform=transform)

        trainloader = torch.utils.data.DataLoader(trainset, shuffle = False, batch_size = len(trainset), pin_memory = using_gpu, drop_last = False)


        file_prefix = 'no_aug'

        if not(args.advanced_augmentation and args.train_aug):
            num_aug = 1

            ann.index_train_add(d, trainloader, index_obj, index_args, use_gpu = using_gpu)


        if args.basic_augmentation:

            file_prefix = 'basic_aug'

            print("ANN: adding augmented trainloaders")

            for i in range(2*pad+1): # x-location of padding for randomcrop
                for j in range(2*pad+1): # y-location of padding for randomcrop
                    for k in range(2): # probability of horizontal flip
                        if i==pad and j==pad and k==0: # same as unaugmented training image 
                            continue
                        transform = transforms.Compose(
                            [transforms.RandomHorizontalFlip(k),
                            transforms.Pad([i,j,2*pad-i,2*pad-j], fill=tuple([min(255, int(round(255 * x))) for x in mu])),
                            transforms.CenterCrop(32),
                            transforms.ToTensor(),
                            transforms.Normalize(mu,sigma)])

                        trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=False, transform=transform)

                        trainloader = torch.utils.data.DataLoader(trainset, shuffle = False, batch_size = len(trainset), pin_memory = using_gpu, drop_last = False )



                        ann.index_add(trainloader, mixup_fn = mixup_fn)

                        num_aug = num_aug + 1
                        print("\t %d, index.ntotal=%d(%d)" %(num_aug, ann.index.ntotal,len(ann.y)))


        if args.advanced_augmentation:

            file_prefix = 'adv_aug'
            if mixup_fn is not None:
                file_prefix = file_prefix + '_mixup'

            auto_augmentation_transforms = [rand_augment_transform(aa_config_string, aa_params)]
            basic_augmentation_transforms = [transforms.RandomCrop(im_dim, padding=4, fill=tuple([min(255, int(round(255 * x))) for x in mu])), transforms.RandomHorizontalFlip()]
            data_transforms = [transforms.ToTensor(), transforms.Normalize(mu,sigma)]
            re_transform = [transforms.RandomErasing(p=reprob, value='random')]

            transform = transforms.Compose(
                auto_augmentation_transforms + 
                basic_augmentation_transforms + 
                data_transforms + 
                re_transform
            )
            print("transform:", transform)
            if args.train_aug:

                file_prefix = file_prefix + '_tr_aug_%d' %args.train_aug

                trainloaders = [trainloader]

                for i in range(args.train_aug):


                    trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=False, transform=transform)

                    trainloader = torch.utils.data.DataLoader(trainset, shuffle = False, batch_size = len(trainset), pin_memory = using_gpu, drop_last = False)

                    trainloaders.append(trainloader)

                num_aug = args.train_aug+1
                ann.index_train_add(d, trainloaders, index_obj, index_args, use_gpu = using_gpu)

            for i in range(args.epochs):

                trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=False, transform=transform)

                trainloader = torch.utils.data.DataLoader(trainset, shuffle = False, batch_size = len(trainset), pin_memory = using_gpu, drop_last = False)

                ann.index_add(trainloader, mixup_fn = mixup_fn)

                num_aug = num_aug + 1
                print("\t %d, index.ntotal=%d(%d)" %(num_aug, ann.index.ntotal,len(ann.y)))

                if not((num_aug)%400 or num_aug==num_epochs):

                    print("Testing after num_aug = ", num_aug)

                    ks  = list(range(1,50)) + list(range(50,100,10)) + list(range(100,1001,100))

                    test_accuracys = np.zeros(len(ks))

                    for ik,k in enumerate(ks):
                        ypred, _, _ = ann.predict(Xtest,k)
                        acc = accuracy(ypred,ytest)
                        test_accuracys[ik] = acc

                    best_acc = np.max(test_accuracys)
                    best_k = ks[np.argmax(test_accuracys)]
                    print("Best acc @ epoch %d = %f(%d)" %(num_aug,best_acc,best_k))

                    search_filename = os.path.join(save_dir,"%s_%s%s_aug%d%s.npy" %(file_prefix,index,index_suffix,num_aug,'_gpu' if using_gpu else ''))
                    search_data = dict(ks = ks, test_accuracys = test_accuracys, best_acc = best_acc, best_k = best_k)
                    np.save(search_filename, search_data)

                    index_filename = os.path.join(save_dir,"%s_%s%s%s.idx" %(file_prefix,index,index_suffix,'_gpu' if using_gpu else ''))
                    if using_gpu:
                        faiss.write_index(faiss.index_gpu_to_cpu(ann.index), index_filename)
                    else:
                        faiss.write_index(ann.index, index_filename)
                    ytr_filename = os.path.join(save_dir,"%s_%s%s%s_ytrain.npy" %(file_prefix,index,index_suffix,'_gpu' if using_gpu else ''))
                    np.save(ytr_filename,ann.y)


        print('Indexing time = ', time.time()-t)

        index_filename = os.path.join(save_dir,"%s_%s%s%s.idx" %(file_prefix,index,index_suffix,'_gpu' if using_gpu else ''))
        if using_gpu:
            faiss.write_index(faiss.index_gpu_to_cpu(ann.index), index_filename)
        else:
            faiss.write_index(ann.index, index_filename)

        ytr_filename = os.path.join(save_dir,"%s_%s%s%s_ytrain.npy" %(file_prefix,index,index_suffix,'_gpu' if using_gpu else ''))
        np.save(ytr_filename,ann.y)

        ks  = list(range(1,50)) + list(range(50,100,10)) + list(range(100,1001,100))
        test_accuracys = np.zeros(len(ks))

        for ik,k in enumerate(ks):
            t = time.time()
            ypred, _, _ = ann.predict(Xtest,k)
            acc = accuracy(ypred,ytest)
            test_accuracys[ik] = acc
            print("%d-NN accuracy = %f (search time = %f)" %(k,acc,time.time()-t))

        best_acc = np.max(test_accuracys)
        best_k = ks[np.argmax(test_accuracys)]
        print("Best acc %f(%d)" %(best_acc,best_k))

        search_filename = os.path.join(save_dir,"%s_%s%s%s.npy" %(file_prefix,index,index_suffix,'_gpu' if using_gpu else ''))
        search_data = dict(ks = ks, test_accuracys = test_accuracys, best_acc = best_acc, best_k = best_k)
        np.save(search_filename, search_data)

        print("Is IVF?: ", index.startswith('ivf'))
        if index.startswith('ivf'):
            ann.index.nprobe = index_args[2]
            test_accuracys_exp = np.zeros(len(ks))
            print("Using nprobe = %d" % index_args[2])
            for ik,k in enumerate(ks):
                t = time.time()
                ypred, _, _ = ann.predict(Xtest,k)
                acc = accuracy(ypred,ytest)
                test_accuracys_exp[ik] = acc
                print("%d-NN accuracy = %f (search time = %f)" %(k,acc,time.time()-t))

            best_acc = np.max(test_accuracys_exp)
            best_k = ks[np.argmax(test_accuracys_exp)]
            print("Best acc %f(%d)" %(best_acc,best_k))

            search_filename = os.path.join(save_dir,"%s_%s%s_nprobe%d%s.npy" %(file_prefix,index,index_suffix,index_args[2],'_gpu' if using_gpu else ''))
            search_data = dict(ks = ks, test_accuracys = test_accuracys_exp, best_acc = best_acc, best_k = best_k)
            np.save(search_filename, search_data)




if __name__ == '__main__':
    main()