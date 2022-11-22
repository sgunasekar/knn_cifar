import faiss
import numpy as np

epoch_size = 50000
num_classes = 10

class ANN():
    def __init__(self):
        pass


    def index_train_add(self, d, trainloaders, index, index_args=(), index_kwargs = {}, use_gpu = False, pca = 0):

        self.index = index(*index_args, **index_kwargs)
        self.use_gpu = use_gpu
        self.d = d
        self.pca = pca

        if use_gpu:
            res = faiss.StandardGpuResources()
            self.cpu_index = self.index
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            print("CPU index ", self.cpu_index, self.cpu_index.d)

        print("ANN: using index ", self.index, self.index.d)

        if not(isinstance(trainloaders,list)):
            trainloaders = [trainloaders]

        X = np.zeros((len(trainloaders)*epoch_size,d)).astype('float32')
        self.y = np.zeros((len(trainloaders)*epoch_size,))
        for i,trainloader in enumerate(trainloaders):

            assert trainloader.batch_size==epoch_size

            Xt,yt = iter(trainloader).next()
            Xt = Xt.numpy().reshape(-1,d).astype('float32')
            yt = yt.numpy()
            X[i*epoch_size:(i+1)*epoch_size,:] = Xt
            self.y[i*epoch_size:(i+1)*epoch_size] = yt

        if self.pca:
            import sklearn.decomposition.PCA as PCA
            self.pca_transform = PCA(n_components = self.pca)
            print("Using PCA on (%d,%d) to reduce dimension to %d" %(X.shape[0],X.shape[1],self.pca))
            X = self.pca_transform.fit_transform(X)
            print("\t New X shape ", X.shape)
            self.d = self.pca

        if not(self.index.is_trained):

            print("\t ANN: training index with %d trainloaders" %len(trainloaders))
            self.index.train(X)

        print("\t ANN: adding %d trainloaders to index" %len(trainloaders))
        self.index.add(X)



    def index_add(self, trainloader, mixup_fn = None):

        assert(self.index.is_trained)

        if mixup_fn is not None and (self.y.ndim<2 or self.y.shape[1]==1):
            num_y= self.y.shape[0]
            y_cls = self.y.astype('int').reshape(-1,1)
            self.y = np.zeros((num_y,num_classes)).astype('float16')
            np.put_along_axis(self.y, y_cls, 1, axis = 1)

            # num_y= self.y.shape[0]
            # one_hot = np.arange(num_classes)
            # self.y = (self.y.reshape((1,num_y,1)) == one_hot.reshape((num_classes,1,1)))
            #self.y = one_hot(torch.tensor(self.y), num_classes, on_value=1, off_value=0, device=device).numpy()


        for X,y in trainloader:

            if mixup_fn is not None:
                X,y = mixup_fn(X,y, device = 'cpu')

            X = X.numpy().reshape(-1,3072).astype('float32')
            if self.pca:
                X = self.pca_transform.transform(X)
            self.y = np.concatenate((self.y, y.numpy().astype('float16')), axis=0)


            self.index.add(X)



    def predict(self, X, k=1, return_knn = True):

        if self.pca:
            X = self.pca_transform.transform(X)

        X = X.astype('float32')
        knn_dist, knn_id = self.index.search(X,k)

        if k==1:

            ypred = self.y[knn_id]
            ypred_out = ypred.squeeze()
            if self.y.ndim > 1 and self.y.shape[1] != 1:
                ypred_out = np.argmax(ypred_out, axis = -1)

        else:

            ypred = self.y[knn_id]
            # print(ypred.shape)
            num_test= ypred.shape[0]

            row_sums = knn_dist.sum(axis=1)
            knn_wt = row_sums[:, np.newaxis]/knn_dist
            knn_wt = knn_wt.reshape(1,num_test,k)

            if ypred.ndim < 3:
                one_hot = np.arange(num_classes)
                one_hot_ypred = (ypred.reshape((1,num_test,k)) == one_hot.reshape((num_classes,1,1))) ## a num_class(c) x num_test(n) x k binary tensor
            ## uses broadcasting rules in numpy
            ## equivalent to ypred -> cxnxk obtained by copying nxk along c and i->cxnxk with i[c']=c'1_{nxk}
            else: #ypred is a n x k x c tensor
                one_hot_ypred = ypred.transpose(2,0,1)


            ypred_out = np.argmax((one_hot_ypred*knn_wt).sum(axis=2).T,axis=1)


        if return_knn:
            return ypred_out, knn_id, knn_dist
        else:
            return ypred_out

        #return ypred

def accuracy(ypred,ytest):

    return (ypred==ytest).mean()