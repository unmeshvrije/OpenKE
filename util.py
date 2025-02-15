import functools
import time
import json

def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print("Finished {} in {} secs".format(repr(func.__name__), round(run_time, 3)))
        return value

    return wrapper


def write_dict_to_json_file(data_dict, file_path):
    if not isinstance(data_dict, dict):
        print("Expecting dict got {type(data_dict)}")
        return
    try:
        with open(file_path, 'w') as f:
            json.dump(data_dict, f, indent=4)
            print(f"Dictionary successfully written to {file_path}")
    except Exception as e:
        print(f"Error writing dictionary to file: {e}")
