Installation instructions
=========================

Requirements
------------

``Deep-Transit`` can take full advantage of the power of GPU. If you don't have a GPU card. 
And it is designed to be CPU compatiable for detection and training.

``Deep-Transit`` has two backends, one is PyTorch and the other is MegEngine. The default is PyTorch backend.
Optionally, if you want to switch to MegEngine backend, you need to install MegEngine with::

    python3 -m pip install megengine -f https://megengine.org.cn/whl/mge.html

You can find more details about MegEngine on its `project <https://github.com/MegEngine/MegEngine>`_.

Installation
------------

To install ``Deep-Transit`` with pip::

    pip install deep-transit

Alternatively you can install it manually::

    git clone https://github.com/ckm3/Deep-Transit.git
    cd Deep-Transit
    python setup.py install

For more info, please open an Issue on the GitHub page.