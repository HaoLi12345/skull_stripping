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
pip install -r /PATH/AGS/src/requirements.txt <br />



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


same for other folders. the order should be matched for nifti files image and label folders <br />
train_set_label_path <br />
├── 1_label.nii.gz <br />
└── 2_label.nii.gz <br />
... <br />
... <br />
... <br />
... <br />
└── 100_label.nii.gz (bunch of nifti files in a folder) <br />





It shows like below during training: <br />

[1,     1] loss: 0.83022  <br />
[1,     2] loss: 0.75864  <br />
[1,     3] loss: 0.63397  <br />
[1,     4] loss: 0.54031  <br />
[1,     5] loss: 0.55116  <br />
[1,     6] loss: 0.59595  <br />
... <br />
... <br />
... <br />

first number is the current epoch number and second number is current iteration number <br />




resume = True ---> continue train from a checkpoint <br />
prefix ---> prefix of saving name. For example, currently in code: prefix='test_123', please also check the folders in ./src/outputFiles <br />







