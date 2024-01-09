[comment]: \page README Readme

# libeisdrt

_libeisdrt_ is a c++ shared library to compute Distribution of Relaxation Times using Tikhonov regularization.
_libeisdrt_ is well integrated to eisgenerator, Eigen and PyTorch.

Full documentaton can be found [here](https://uvos.xyz/kiss/libdrtdoc) or by building the doc target

## Building

the main devolpment platform of _libeisdrt_ is linux, and _libeisdrt_ works best on UNIX-like systems, however _libeisdrt_ is fully portable and should compile on any platform that supports the requriements below

### Requirments
- [git](https://git-scm.com/) is requried to get the source
- a C++20 compliant compiler like [gcc](https://gcc.gnu.org/)
- [cmake](https://cmake.org/) version 3.19 or later
- [libtorch](https://pytorch.org/get-started/locally/) is optional, but is required for libtorch Tensor support
- [eisgenerator](https://git-ce.rwth-aachen.de/carl_philipp.klemm/eisgenerator) is optional, but is required for eisgenerator DataPoint support
- [PkgConfig](https://www.freedesktop.org/wiki/Software/pkg-config/) is optional, but is required for eisgenerator support
- [eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) version 4.0 or later

### Procedure
```
$ git clone https://git-ce.rwth-aachen.de/carl_philipp.klemm/libdrt
$ mkdir libdrt/build
$ cd libdrt/build
$ cmake ..
$ make
$ make install
$ make doc
```

# Licence

_libeisdrt_ is licenced to you under the LGPL version 3 , or (at your option) any later version. see lgpl-3.0.txt or [LGPL](https://www.gnu.org/licenses/lgpl-3.0.en.html) for details
