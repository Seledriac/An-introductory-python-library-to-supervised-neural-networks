
* If you want to try the hd_recognition test script :
    - open shell
    - cd <your_path_to_the_library/hd_recognition>
    - python test_hd.py

* If you want to import the network module and make your own tests in the python shell
    - open shell
    ```
    python
    import os
    os.chdir("<your_path_to_the_library>")
    ```
    * To check if you are in the right virtual python emplacement --> "os.getcwd()"
    * There you can "import network", build/train a model on your own training data, save it, and use it to make predictions

* (Coming soon) Use the library with a graphical interface 

