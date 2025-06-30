# Reinforcemnt Learning (based on the book Understanding Deep Learning by Simon J.D. Prince[[1]](#references))
This repository is based on a group project in the course Understanding Deep Learning at University of Osnabrück.

## Overview
Reinforcement Learning is a 

## QuickStart
To start of quickly and be able to execute the code properly, follow this guide.
For fastest usage, go to this [jupyter notebook](https://colab.research.google.com/drive/1ae7qFCyGvhH7TT2yE0qTrL9GH5_HAyJ9?usp=sharing) in google collab.


Fist we need to install [Git](#git) to be able to clone this repository.
Then decide, whether you want to set up your virtual environment with [venv](#venv) (built into Python) or [Conda](#conda) (a package and environment manager from Anaconda/Miniconda).

### Install Git
<a name="git"></a>
Download and install Git:

- Visit the [official Git website](https://git-scm.com/) to download the latest version of Git.
- Follow the installation instructions for your operating system.

### Clone the Git Repository

- Open a terminal or command prompt.
- Go to the directory where you want to store everything regarding the course:
```bash
cd <directory_name>
```
- Clone the Git repository:
```bash
git clone https://github.com/lahellmann/UDL-Reinforcement-Learning
```
- Change into the cloned repository:
```bash
cd UDL-Reinforcement-Learning
```

### Set Up a Virtual Environment (venv)
<a name="venv"></a>

Download and install Python:
- Visit the [official Python website](https://www.python.org/) to download the latest version of Python.
- During installation, make sure to check the option that adds Python to your system's PATH.

- Create a virtual environment:
```bash 
python -m venv venv
```
- Activate the virtual environment:
--> On Windows:
```bash
.\venv\Scripts\activate
```
--> On Unix or MacOS:
```bash
source venv/bin/activate
```
- Install required packages
```bash
pip install -r requirements.txt
```

### Set Up a Virtual Environment (conda)
<a name="conda"></a>
- Create a virtual environment:
1. Open your terminal (Command Prompt on Windows, Terminal on macOS/Linux).
2. Navigate to the directory where you saved the environment.yml file. (This should be YOUR_PATH/OED_Game_Theoretic_Framework_MBRL/ )
3. Execute the following command to create the environment:

```bash 
conda create -m venv -f environment.yml
```
- Activate the virtual environment:
--> On Windows, Unix and MacOS:
```bash
conda activate venv
```
- Install required packages
```bash
pip install -r requirements.txt
```

## How to use this repository


## References
<a name="references"></a>
[1] S. J. D. Prince, Understanding Deep Learning. The MIT Press, 2023. [webiste to book](https://git-scm.com/](https://udlbook.github.io/udlbook/)
