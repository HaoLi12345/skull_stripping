# Human brain extraction with deep learning

```
@inproceedings{li2022human,
  title={Human brain extraction with deep learning},
  author={Li, Hao and Zhu, Qibang and Hu, Dewei and Gunnala, Manasvi R and Johnson, Hans and Sherbini, Omar and Gavazzi, Francesco and D’Aiello, Russell and Vanderver, Adeline and Long, Jeffrey D and others},
  booktitle={Medical Imaging 2022: Image Processing},
  volume={12032},
  pages={369--375},
  year={2022},
  organization={SPIE}
}
```

installation (need to create a new environment if you don't want to install those in base) <br />
pip install -r /PATH/requirements.txt <br />



Change paths for train and validation sets, and it is ready to run. <br />
example: <br />
train_set_image_path <br />
├── 1.nii.gz <br />
└── 2.nii.gz <br />
... <br />
... <br />
... <br />
... <br />
└── 100.nii.gz (bunch of nifti files in a folder) <br />


same for other folders. the order should be matched for image and label folder
train_set_label_path
├── 1_label.nii.gz
└── 2_label.nii.gz
...
...
...
...
└── 100_label.nii.gz (bunch of nifti files in a folder)





It shows like below during training:

[1,     1] loss: 0.83022 
[1,     2] loss: 0.75864 
[1,     3] loss: 0.63397 
[1,     4] loss: 0.54031 
[1,     5] loss: 0.55116 
[1,     6] loss: 0.59595 
...
...
...

first number is the current epoch number and second number is current iteration number




resume = True ---> continue train from a checkpoint
prefix ---> prefix of saving name. For example, currently in code: prefix='test_123', please also check the folders in ./src/outputFiles







