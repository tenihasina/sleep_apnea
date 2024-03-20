# __init__.py

# Define package-level variables or constants
PACKAGE_NAME = "sleep_apnea"

# Import modules or sub-packages to make them accessible from the package namespace
from .data_augment import *
from .main import *
from .utils import *
from .data_processing import *
from .model import *