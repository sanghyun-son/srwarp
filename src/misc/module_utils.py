import sys
import inspect
import importlib

def load_with_exception(module, name):
    try:
        m = importlib.import_module('.' + module, package=name)
    except ModuleNotFoundError as e:
        print('Error: ' + module + ' is not defined well!')
        print(e)
        sys.exit(1)
    except Exception as e:
        print(e)
        sys.exit(1)

    return m

def find_representative(m):
    target_class = None
    # If the module has a representative class
    if hasattr(m, 'REPRESENTATIVE'):
        target_class = getattr(m, 'REPRESENTATIVE')
    # Otherwise pick one in the top
    else:
        for name, obj in inspect.getmembers(m):
            if inspect.isclass(obj):
                target_class = getattr(m, name)
                break

    return target_class
