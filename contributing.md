# Contributing to inequality

Contributions to inequality are much appreciated.

## Steps to Contribute

1. Fork the inequality git repository
2. Create a development environment
3. Activate the new environment
4. Install project dependencies
5. Verify installation and run tests
6. Build documentation
7. Submitting a Pull Request

## 1. Fork the inequality git repository

- On github, fork the repository at: <https://github.com/pysal/inequality>
- From your new fork, grab the clone url:
```
git clone git@github.com:your-user-name/inequality.git inequality-yourname
cd inequality-yourname
git remote add upstream git://github.com/pysal/inequality.git
```

## 2. Create a development environment

- Install either [Anaconda](http://docs.continuum.io/anaconda/)  or [miniconda](http://conda.pydata.org/miniconda.html)
- `cd` into the `inequality-yourname` source directory that you cloned in step 1

```
conda create --name inequality python=3.10
```

## 3. Activate the new environment
```
conda activate inequality
```
## 4. Install project dependencies
```
pip install .[dev,docs,tests]
```

## 5. Verify installation and run tests
```
python -c "import libpysal; print('libpysal version:', libpysal.__version__)"
pytest
```


## 6. Build documentation
```
cd docs
make html
```

## 7. Submitting a Pull Request

If you have made changes that you have pushed to your forked repository, you can
submit a pull request to have them integrated into the `inequality` code base.

See the [GitHUB tutorial](https://help.github.com/articles/using-pull-requests/).
