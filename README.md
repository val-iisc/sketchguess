# sketchguess
Repository for code, models and dataset for the paper Game of Sketches: Deep Recurrent Models of Pictionary-style Word Guessing accepted at 36th AAAI Conference on Artificial Intelligence (AAAI-18), New Orleans, USA.


### License

This code is released under the MIT License (Please refer to the LICENSE file for details).

### Citation
Please cite our paper in your publications if it helps your research:
    
    
    @article{,
    Author = {Sarvadevabhatla,Ravi Kiran and Surya, Shiv and Mittal, Trisha and
    Babu R, Venkatesh},
        Title = {Game of Sketches: Deep Recurrent Models of Pictionary-style Word Guessing},
        Journal = {ArXiv e-prints},
        eprint = {},
        Keywords = {Computer Science - Computer Vision and Pattern Recognition},
        Year = {2018},
        Month = {February},
       }
<!---
    @inproceedings{,
        Author = {},
        Title = {},
        Booktitle = {},
        Year = {2018}
    }
--->
### Dependencies and Installation

1. Code for Sketchguess is based on Lasagne\Theano. This code was tested on UBUNTU 14.04 on the folowing NVIDIA GPUs: NVIDIA TITAN X.

2. Download CNN features that are input to the LSTM, Word2Vec dicitonary and dataset,

   Instructions are available [here](https://github.com/val-iisc/sketchguess/blob/master/data/README.md).

3. To test on the Sketchguess recurrent on trained model:
  
   ```bash
   $ git clone https://github.com/val-iisc/sketchguess.git 
   $ bash models/download_models.sh
   $ bash src/run_model.sh
   ```
4. The predictions of the Sketchguess recurrent model can be viewed in the log file : out/machine_generated_guesses.log .

5. The embedding representations regressed by the lstm model are stored in : out/pred_release.npz . This file is used to generate the      machine guesses via a KNN-search of "data/all_w2v.mat".



### Q&A
1. If automatic download of pre-trained models fail, you can find instructions for manual download [here](https://github.com/val-iisc/sketchguess/blob/master/models/README.md).

2. Please send message to ravika@gmail.com or shiv.surya314@gmail.com if you have any query regarding the code.
