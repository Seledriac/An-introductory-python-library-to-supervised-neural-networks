* If you want to try the compiled test script :
    - open cmd
    ```
    cd <your_windows_path_to_the_Python_scripts_package_with_simple_backslashes>
    python test.py
    ```
* If you want to try the compiled test script :
    - open cmd
    - cd <your_windows_path_to_the_Python_scripts_package_with_simple_backslashes>
    - python test_hd.py

* If you want to import the network module and make your own tests in the python shell
    - open python in cmd
    ```Python
    import os
    os.chdir("<your_windows_path_to_the_Python_scripts_package_with_double_backslashes>")
    ```
    - example : `os.chdir("C:\\Users\\boumb\\OneDrive\\Bureau\\IA\\Neural-Networks\\tmp_weights")`
    
    * To check if you are in the right virtual python emplacement --> "os.getcwd()"
    * There you can "import network", try executing the "test" scripts once ("import test"), and build/train a model, before using it to make predictions

* (Coming soon) Use the library with a graphical interface 

