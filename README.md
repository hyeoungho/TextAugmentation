# UIFtextaugmentation
UIF Text Augmentation Project

**Prerequisite**

I assume that the development environment is on Windows. Due to the limitation of CUDA support on WSL2, I struggled two days but abandoned the idea of using Linux environment for this development (though I prefer that).
Under the assumption, you want to install below things first:
  1) Anaconda with Python 3.7.* (Please note that we need to have the same version of python)
  2) Spyder (it will come with Anaconda)
  3) Virtual environment (all the required packages are listed in requirement.txt)
    On your anaconda shell (and of course on your cloned directory root):
    >python -m venv .venv
    >.venv\Script\activate
    [You should be able to see (.venv) starting from the next line on your shell]
    Then you play with pip to install all the packages that you need to install

**Data**

  Our data can only be shared internally. Please DO NOT include it in your PR.

**Pretrained bert model**

  You will find where to download the trained model in the folder. 

**Pull Request**

  Only add code changes on your PR.
