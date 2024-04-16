# Section 3 Jupyter Overview 

- Skipping the first two but as I'm just going to use vscode.
- I am going to do the virtual environment one though.

# 7 Virtual Environments

- If you don't specify a version of Python, your mchns version will be used
- 

```
## example: 
## conda create --name [your_env_name] [libs]

conda create --name snowflakes biopythony
conda activate snowflakes
conda decativate 
```

## Create an ENV with python 3.5 and numpy

```
conda create --name python35 python=3.5 numpy 
```

## list envs
```
conda info --envs
```

## list env info
```
conda info

     active environment : python35
    active env location : /home/gil/miniconda3/envs/python35
            shell level : 2
       user config file : /home/gil/.condarc
 populated config files : 
          conda version : 24.3.0
    conda-build version : not installed
         python version : 3.12.1.final.0
                 solver : libmamba (default)
       virtual packages : __archspec=1=skylake
                          __conda=24.3.0=0
                          __cuda=12.2=0
                          __glibc=2.35=0
                          __linux=6.5.0=0
                          __unix=0=0
       base environment : /home/gil/miniconda3  (writable)
      conda av data dir : /home/gil/miniconda3/etc/conda
  conda av metadata url : None
           channel URLs : https://repo.anaconda.com/pkgs/main/linux-64
                          https://repo.anaconda.com/pkgs/main/noarch
                          https://repo.anaconda.com/pkgs/r/linux-64
                          https://repo.anaconda.com/pkgs/r/noarch
          package cache : /home/gil/miniconda3/pkgs
                          /home/gil/.conda/pkgs
       envs directories : /home/gil/miniconda3/envs
                          /home/gil/.conda/envs
               platform : linux-64
             user-agent : conda/24.3.0 requests/2.31.0 CPython/3.12.1 Linux/6.5.0-26-generic ubuntu/22.04.4 glibc/2.35 solver/libmamba conda-libmamba-solver/23.12.0 libmambapy/1.5.3
                UID:GID : 1000:1000
             netrc file : None
           offline mode : False

```

## list the packages installed in the environment
```
conda list 
```

## Add a package to an environment 
```
# conda install -n <env_name> <package> 
```