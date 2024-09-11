## Albumy

*Capture and share every wonderful moment.*

> Example application for *[Python Web Development with Flask](https://helloflask.com/en/book/1)* (《[Flask Web 开发实战](https://helloflask.com/book/1)》).

Demo: http://albumy.helloflask.com

![Screenshot](https://helloflask.com/screenshots/albumy.png)

## Installation

clone:
```
$ git clone https://github.com/greyli/albumy.git
$ cd albumy
```
create & activate virtual env then install dependency:

with venv/virtualenv + pip:
```
$ python -m venv env  # use `virtualenv env` for Python2, use `python3 ...` for Python3 on Linux & macOS
$ source env/bin/activate  # use `env\Scripts\activate` on Windows
$ pip install -r requirements.txt
```
or with Pipenv:
```
$ pipenv install --dev
$ pipenv shell
```
generate fake data then run:
```
$ flask forge
$ flask run
* Running on http://127.0.0.1:5000/
```
Test account:
* email: `admin@helloflask.com`
* password: `helloflask`

## Generate Azure API key and Endpoint
1) Obtain API key and endpoint by following this link (steps 1-3): https://github.com/mlip-cmu/f2024/blob/main/labs/lab01.md#connecting-to-the-azure-vision-api
2) Create config.json within the "albumy" folder. Make sure it is in the same level as blueprints, forms, statics, templates, etc.
3) add into config.json:
```
{
   "AZURE_SUBSCRIPTION_KEY": "replace_with_your_api_key",
   "AZURE_ENDPOINT_URL": "replace_with_your_endpoint"
}
```

## License

This project is licensed under the MIT License (see the
[LICENSE](LICENSE) file for details).
