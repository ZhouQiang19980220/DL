"""
Some utility functions for deep learning
"""
from .packages import *

def get_cuda(idx: int = 0) -> torch.device:
    """
    Get the cuda device
    """
    if torch.cuda.is_available():
        if idx < torch.cuda.device_count():
            return torch.device(f'cuda:{idx}')
        else:
            raise ValueError(f'cuda:{idx} is not available, available devices are: {torch.cuda.device_count()}')
    else:
        raise ValueError('cuda is not available')
    
def get_mps() -> torch.device:
    """
    Get the aplle mps device
    """
    if torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        raise ValueError('mps is not available')
    
def get_cpu() -> torch.device:
    """
    Get the cpu device
    """
    return torch.device('cpu')
    
def get_cuda_cpu(cuda_idx: int = 0) -> torch.device:
    """
    Get the cuda device if available else get the cpu device
    """
    try :
        return get_cuda(cuda_idx)
    except ValueError as e:
        logger.error(e)
        get_cpu()

def get_mps_cpu() -> torch.device:
    """
    Get the mps device if available else get the cpu device
    """
    try :
        return get_mps()
    except ValueError as e:
        logger.error(e)
        get_cpu()

def get_cuda_mps_cpu() -> torch.device:
    """
    Get the cuda device if available else get the mps device if available else get the cpu device
    """
    try :
        return get_cuda()
    except ValueError as e:
        logger.error(e)
        try :
            return get_mps()
        except ValueError as e:
            logger.error(e)
            get_cpu()

#%% initialize 2d list
def get_2d_list(rows: int, cols: int, val: Any=None) -> List[List[Any]]:
    """
    Get a 2D list of size rows x cols
    """
    return [[val for _ in range(cols)] for _ in range(rows)]

#%% open or close matplotlib interactive mode
def set_interactive(interactive: bool = True):
    """
    Set the matplotlib interactive mode
    """
    if interactive:
        plt.ion()
    else:
        plt.ioff()

def open_interactive():
    """
    Open the matplotlib interactive mode
    """
    set_interactive(True)

def close_interactive():
    """
    Close the matplotlib interactive mode
    """
    set_interactive(False)

#%% get file dir
def get_file_dir(file: str) -> str:
    """
    Get the directory of the file
    """
    return os.path.dirname(os.path.abspath(file))

def get_parent_dir(file: str) -> str:
    """
    Get the parent directory of the file
    """
    return os.path.dirname(get_file_dir(file))

#%% imshow imgs
def imshow(
        imgs: Sequence[Union[np.ndarray, Image.Image]], 
        titles: Optional[Sequence[str]] = None,
        cmap: str = 'gray',
        rows = None, 
        cols = None,
        sacle = 1.0,):
    """
    display multiple images
    """
    if isinstance(imgs, (list, tuple)):
        if len(imgs) == 0:
            raise ValueError('imgs is empty')
        if not all(isinstance(img, (np.ndarray, Image.Image)) for img in imgs):
            raise ValueError('all elements in imgs should be numpy array or PIL Image')
    if isinstance(imgs, (np.ndarray, Image.Image)):
        imgs = [imgs]
    if titles is None:
        titles = [f'img{i}' for i in range(len(imgs))]
    if len(imgs) > len(titles):
        raise ValueError('titles should have at least the same number of elements as imgs')
    if rows is None and cols is None:
        rows = 1
        cols = len(imgs)

    figsize = (cols * sacle, rows * sacle)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    
    axes = axes.flatten()

    for i, (img, title) in enumerate(zip(imgs, titles)):
        ax = axes[i]
        ax.imshow(img, cmap=cmap)
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()