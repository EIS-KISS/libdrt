# libeisdrt

_libeisdrt_ is a c++ shared library to compute Distribution of Relaxation Times using Tikhonov regularization.
_libeisdrt_ is well integrated to eisgenerator, Eigen and PyTorch.

This manual is divided in the following sections depending on what datatypes you want to use libeisdrt with:
- \ref EIGENAPI api to use in [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) applications
- \ref TORCHAPI api to use in [libtorch/PyTorch](https://pytorch.org/) applications
- \ref EISAPI  api to use in [eisgenerator](https://git-ce.rwth-aachen.de/carl_philipp.klemm/eisgenerator) applications
- \ref TYPES types used by all apis


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
