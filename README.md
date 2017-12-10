Code uses imports relative to ```<project_path>/src```
You need to set **PYTHONPATH** environment variable first.
```bash
export PYTHONPATH=<project_path>/src
docker build . --tag instronizer
docker run -p 80:80 --name instronizer_container instronizer
```
