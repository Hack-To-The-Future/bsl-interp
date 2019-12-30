# BSL Interpertor using Tensorflow

## Pre-requisites:

Once you've cloned this git repository, `cd` into the root directory and create a new python virtual environment. The following instructions are for Windows:

```
~\AppData\Local\Programs\Python\Python37\python.exe -m pip install --upgrade pip
~\AppData\Local\Programs\Python\Python37\python.exe -m pip install --upgrade virtualenv
~\AppData\Local\Programs\Python\Python37\python.exe -m virtualenv -p python3.7 bsl-interp
```

Now activate your virtual environment, in Windows that would look like this:

```
.\bsl-interp\Scripts\activate.ps1
```

You can check this worked by making sure that your `sys` prefix points to the virtual environment diretory just created:

```
python -c "import sys; print(sys.prefix)"
```

Now install all of the dependancies:

```
python -m pip install --upgrade -r .\requirements.txt
```

This will take a few minutes, make sure that there aren't any errors produced from the installation.

