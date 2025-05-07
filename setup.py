# seteup.py : 
# This setup.py is responsibile in creating my machine learning application as a package. 
# with the help of setup.py i will be able to build my entire machine learning application as a package and even deploy in pypi and from pypi anybody can do installation this setup.py and anybody can also use. 
# so this is the reason we will use this setup.py file
# ---> in shot building our application as a package itself.

from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT='-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    this function will return the list of requirements
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements



setup(
name='mlproject',
version='0.0.1', #whenever my next version will come i will keep on updateing this and automatically the entire packages will be built and it will be go into the pypi.
author='Naveen',
author_email='naveenchawla1994@gmail.com',
packages=find_packages(),
#install_requires=['pandas','numpy','seaborn','matplotlib'], #automatically it will do the install of the all the libraries.
#scenario : if we want 100s of pacakages installation at a time then we will go for another method.
#we will try to create a function and that function i will basically give my path over here
install_requires=get_requirements('requirements.txt') #it should be able to read the all the files



)