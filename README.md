This Code is based python3.0
This Code represent a super-resolution method for hyperspectral image

How to use it?
Open the CMD command under the path of main.py
and then execute train_srresnet.sh, train_srganc.sh, test_srganc.sh in turn according to the training order. 
Open these files with note-book, copy the part after "Python" to CMD, and press enter to execute.
We prepare the trainging data and test data in the from of matlab,and is from the Hyper-spectral image:Washington DC mall,
the data is under the path "data",and well fromed.

How to deal with the settings?
ALL the parameters that you need to set is in the main.py, according your training needs change the variable "Flags"

Parameter Settings Functions:
(1）3DSRResnet Model:
These following variables are need to be set:
    Out_putdir model:outputs location and filename, default current directory
    The summary_dir:training procedure log store, preferably the same as output_dir, exists by default under the log of output_dir
    Task: SRResnet
    Batch_size:does not need to be ignored. The default is 1 time and 1 picture
    Num_resblock:it is recommended to be less than or equal to 8
    The learning_rate: this variable is adjustable, suggesting 0.00005
    The max_iter: the maximum number of iterations is about 10w
    Save_freq:training results are stored at a frequency of one model every N times
    channel_num: the number of the channel in the data (DC set is 210)
(2)3DSRGAN:
    Out_putdir and summary_dir are not the same as SRResnet
    pre_trained_model_type: if ti is "SRResnet", which means to train with the trained SRResnet; if it is None, it means to train SRGAN from scratch
    Pre_trained_model is set to True if pre_trained_model_type is SRResnet
    Task: SRGAN
    Check_point:if you use SRResnet to train SRGAN, checkpoint is the SRResnet model location such as./0909_SRResnet_01/model-43000, note the path containing the model name
    Other settings is present in 3DSRResnet
(3)test:
    Out_putdir and summary_dir are where the results are
    Just notice that "num_resblock" is the same as before
    Checkpoint: the location and name of the model you want to test
    
 After successful execution, there is a result.mat file under output_dir, which contains the original image OR of the test, the real HR image and the image LR of the SR. You can use matlab to execute result.m to obtain the files that ENVI can open.
