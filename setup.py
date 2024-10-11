from setuptools import find_packages,setup
'''
def get_requirements(file_path):
    requirements =[]
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements] 
        
    return requirements
'''
setup(
    name = "ML_project",
    version = "0.0.1",
    author="Reyan",
    author_email="reyanalam115@gmail.com",
    packages=find_packages(),
    install_requires=['pandas','numpy','matplotlib'],
    #when we need multiple packages , writing name is not feasible. so we create a func
)
