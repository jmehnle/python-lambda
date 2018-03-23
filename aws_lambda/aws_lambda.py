# -*- coding: utf-8 -*-
from __future__ import print_function

import hashlib
import json
import logging
import os
import packaging.requirements
import requests
import sys
import tempfile
import time
import zipfile

from collections import defaultdict
from imp import load_source
from shutil import copy
from shutil import copyfile
from shutil import copytree

import boto3
import botocore
import pip
import yaml

from .helpers import *


ARN_PREFIXES = {
    'us-gov-west-1': 'aws-us-gov',
}

log = logging.getLogger(__name__)


def cleanup_old_versions(
    src, keep_last_versions,
    config_file='config.yaml', profile_name=None,
):
    """Deletes old deployed versions of the function in AWS Lambda.

    Won't delete $Latest and any aliased version

    :param str src:
        The path to your Lambda ready project (folder must contain a valid
        config.yaml and handler module (e.g.: service.py).
    :param int keep_last_versions:
        The number of recent versions to keep and not delete
    """
    if keep_last_versions <= 0:
        print("Won't delete all versions. Please do this manually")
    else:
        cfg = read_cfg(src, config_file, profile_name)

        profile_name = cfg.get('profile')
        aws_access_key_id = cfg.get('aws_access_key_id')
        aws_secret_access_key = cfg.get('aws_secret_access_key')

        client = get_client(
            'lambda', profile_name, aws_access_key_id, aws_secret_access_key,
            cfg.get('region'),
        )

        response = client.list_versions_by_function(
            FunctionName=cfg.get('function_name'),
        )
        versions = response.get('Versions')
        if len(response.get('Versions')) < keep_last_versions:
            print('Nothing to delete. (Too few versions published)')
        else:
            version_numbers = [elem.get('Version') for elem in
                               versions[1:-keep_last_versions]]
            for version_number in version_numbers:
                try:
                    client.delete_function(
                        FunctionName=cfg.get('function_name'),
                        Qualifier=version_number,
                    )
                except botocore.exceptions.ClientError as e:
                    print('Skipping Version {}: {}'
                          .format(version_number, e.message))


def deploy(
        src, use_requirements=False, local_package=None,
        config_file='config.yaml', profile_name=None,
):
    """Deploys a new function to AWS Lambda.

    :param str src:
        The path to your Lambda ready project (folder must contain a valid
        config.yaml and handler module (e.g.: service.py).
    :param str local_package:
        The path to a local package with should be included in the deploy as
        well (and/or is not available on PyPi)
    """
    # Load and parse the config file.
    cfg = read_cfg(src, config_file, profile_name)

    # Copy all the pip dependencies required to run your code into a temporary
    # folder then add the handler file in the root of this directory.
    # Zip the contents of this folder into a single file and output to the dist
    # directory.
    path_to_zip_file = build(
        src, config_file=config_file,
        use_requirements=use_requirements,
        local_package=local_package,
    )

    if function_exists(cfg, cfg.get('function_name')):
        update_function(cfg, path_to_zip_file)
    else:
        create_function(cfg, path_to_zip_file)


def deploy_s3(
    src, use_requirements=False, local_package=None,
    config_file='config.yaml', profile_name=None,
):
    """Deploys a new function via AWS S3.

    :param str src:
        The path to your Lambda ready project (folder must contain a valid
        config.yaml and handler module (e.g.: service.py).
    :param str local_package:
        The path to a local package with should be included in the deploy as
        well (and/or is not available on PyPi)
    """
    # Load and parse the config file.
    cfg = read_cfg(src, config_file, profile_name)

    # Copy all the pip dependencies required to run your code into a temporary
    # folder then add the handler file in the root of this directory.
    # Zip the contents of this folder into a single file and output to the dist
    # directory.
    path_to_zip_file = build(
        src, config_file=config_file, use_requirements=use_requirements,
        local_package=local_package,
    )

    use_s3 = True
    s3_file = upload_s3(cfg, path_to_zip_file, use_s3)
    if function_exists(cfg, cfg.get('function_name')):
        update_function(cfg, path_to_zip_file, use_s3, s3_file)
    else:
        create_function(cfg, path_to_zip_file, use_s3, s3_file)


def upload(
        src, use_requirements=False, local_package=None,
        config_file='config.yaml', profile_name=None,
):
    """Uploads a new function to AWS S3.

    :param str src:
        The path to your Lambda ready project (folder must contain a valid
        config.yaml and handler module (e.g.: service.py).
    :param str local_package:
        The path to a local package with should be included in the deploy as
        well (and/or is not available on PyPi)
    """
    # Load and parse the config file.
    cfg = read_cfg(src, config_file, profile_name)

    # Copy all the pip dependencies required to run your code into a temporary
    # folder then add the handler file in the root of this directory.
    # Zip the contents of this folder into a single file and output to the dist
    # directory.
    path_to_zip_file = build(
        src, config_file=config_file, use_requirements=use_requirements,
        local_package=local_package,
    )

    upload_s3(cfg, path_to_zip_file)


def invoke(
    src, event_file='event.json',
    config_file='config.yaml', profile_name=None,
    verbose=False,
):
    """Simulates a call to your function.

    :param str src:
        The path to your Lambda ready project (folder must contain a valid
        config.yaml and handler module (e.g.: service.py).
    :param str alt_event:
        An optional argument to override which event file to use.
    :param bool verbose:
        Whether to print out verbose details.
    """
    # Load and parse the config file.
    cfg = read_cfg(src, config_file, profile_name)

    # Set AWS_PROFILE environment variable based on `--profile` option.
    if profile_name:
        os.environ['AWS_PROFILE'] = profile_name

    # Load environment variables from the config file into the actual
    # environment.
    env_vars = cfg.get('environment_variables')
    if env_vars:
        for key, value in env_vars.items():
            os.environ[key] = get_environment_variable_value(value)

    # Load and parse event file.
    path_to_event_file = os.path.join(src, event_file)
    event = read(path_to_event_file, loader=json.loads)

    # Tweak to allow module to import local modules
    try:
        sys.path.index(src)
    except ValueError:
        sys.path.append(src)

    handler = cfg.get('handler')
    # Inspect the handler string (<module>.<function name>) and translate it
    # into a function we can execute.
    fn = get_callable_handler_function(src, handler)

    # TODO: look into mocking the ``context`` variable, currently being passed
    # as None.

    start = time.time()
    results = fn(event, None)
    end = time.time()

    print('{0}'.format(results))
    if verbose:
        print('\nexecution time: {:.8f}s\nfunction execution '
              'timeout: {:2}s'.format(end - start, cfg.get('timeout', 15)))


def init(src, minimal=False):
    """Copies template files to a given directory.

    :param str src:
        The path to output the template lambda project files.
    :param bool minimal:
        Minimal possible template files (excludes event.json).
    """

    templates_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'project_templates',
    )
    for filename in os.listdir(templates_path):
        if (minimal and filename == 'event.json') or filename.endswith('.pyc'):
            continue
        dest_path = os.path.join(templates_path, filename)

        if not os.path.isdir(dest_path):
            copy(dest_path, src)


def build(
    src, use_requirements=False, local_package=None,
    config_file='config.yaml', profile_name=None,
):
    """Builds the file bundle.

    :param str src:
       The path to your Lambda ready project (folder must contain a valid
        config.yaml and handler module (e.g.: service.py).
    :param str local_package:
        The path to a local package with should be included in the deploy as
        well (and/or is not available on PyPi)
    """
    # Load and parse the config file.
    cfg = read_cfg(src, config_file, profile_name)

    # Get the absolute path to the output directory and create it if it doesn't
    # already exist.
    dist_directory = cfg.get('dist_directory', 'dist')
    path_to_dist = os.path.join(src, dist_directory)
    mkdir(path_to_dist)

    # Combine the name of the Lambda function with the current timestamp to use
    # for the output filename.
    function_name = cfg.get('function_name')
    output_filename = '{0}-{1}.zip'.format(timestamp(), function_name)

    path_to_temp = tempfile.mkdtemp(prefix='aws-lambda')
    pip_install_to_target(
        cfg,
        path_to_temp,
        use_requirements=use_requirements,
        local_package=local_package,
    )

    # Hack for Zope.
    if 'zope' in os.listdir(path_to_temp):
        print(
            'Zope packages detected; fixing Zope package paths to '
            'make them importable.',
        )
        # Touch.
        with open(os.path.join(path_to_temp, 'zope/__init__.py'), 'wb'):
            pass

    # Gracefully handle whether ".zip" was included in the filename or not.
    output_filename = (
        '{0}.zip'.format(output_filename)
        if not output_filename.endswith('.zip')
        else output_filename
    )

    # Allow definition of source code directories we want to build into our
    # zipped package.
    build_config = defaultdict(**cfg.get('build', {}))
    build_source_directories = build_config.get('source_directories', '')
    build_source_directories = (
        build_source_directories
        if build_source_directories is not None
        else ''
    )
    source_directories = [
        d.strip() for d in build_source_directories.split(',')
    ]

    files = []
    for filename in os.listdir(src):
        if os.path.isfile(filename):
            # Ignore Mac OS cache files and Vim swap files:
            if filename == '.DS_Store' or re.search(r'^\..*\.sw.$', filename):
                continue
            if filename == config_file:
                continue
            print('Bundling: %r' % filename)
            files.append(os.path.join(src, filename))
        elif os.path.isdir(filename) and filename in source_directories:
            print('Bundling directory: %r' % filename)
            files.append(os.path.join(src, filename))

    # "cd" into `temp_path` directory.
    os.chdir(path_to_temp)
    for f in files:
        if os.path.isfile(f):
            _, filename = os.path.split(f)

            # Copy handler file into root of the packages folder.
            copyfile(f, os.path.join(path_to_temp, filename))
        elif os.path.isdir(f):
            destination_folder = os.path.join(path_to_temp, f[len(src) + 1:])
            copytree(f, destination_folder)

    # Zip them together into a single file.
    # TODO: Delete temp directory created once the archive has been compiled.
    path_to_zip_file = archive('./', path_to_dist, output_filename)
    return path_to_zip_file


def get_callable_handler_function(src, handler):
    """Tranlate a string of the form "module.function" into a callable
    function.

    :param str src:
      The path to your Lambda project containing a valid handler file.
    :param str handler:
      A dot delimited string representing the `<module>.<function name>`.
    """

    # "cd" into `src` directory.
    os.chdir(src)

    module_name, function_name = handler.split('.')
    filename = get_handler_filename(handler)

    path_to_module_file = os.path.join(src, filename)
    module = load_source(module_name, path_to_module_file)
    return getattr(module, function_name)


def get_handler_filename(handler):
    """Shortcut to get the filename from the handler string.

    :param str handler:
      A dot delimited string representing the `<module>.<function name>`.
    """
    module_name, _ = handler.split('.')
    return '{0}.py'.format(module_name)


def _install_packages(path, packages, python_runtime):
    """Install all packages listed to the target directory.

    Ignores any package that includes Python itself and python-lambda as well
    since its only needed for deploying and not running the code

    :param str path:
        Path to copy installed pip packages to.
    :param list packages:
        A list of packages to be installed via pip.
    """
    #def _filter_blacklist(package):
    #    blacklist = ['-i', '#', 'Python==', 'python-lambda==']
    #    return all(package.startswith(entry) is False for entry in blacklist)
    #packages = filter(_filter_blacklist, packages)
    for package, req in packages.items():
        #if package.startswith('-e '):
        #    package = package.replace('-e ', '')

        try:
            wheel_path = get_cached_manylinux_wheel(package, list(req.specifier)[0].version, python_runtime)
            print('Installing {} from wheel'.format(package))
            #shutil.rmtree(os.path.join(path, package), ignore_errors=True)
            with zipfile.ZipFile(wheel_path) as zfile:
                zfile.extractall(path)
        except ValueError as e:
            # No such wheel, install via pip:
            print('Installing {} via pip'.format(package))
            os.environ['RADIX_NO_EXT'] = '1'
            pip.main(['install', package, '-t', path, '--ignore-installed'])


def pip_install_to_target(cfg, path, use_requirements=False, local_package=None):
    """For a given active virtualenv, gather all installed pip packages then
    copy (re-install) them to the path provided.

    :param str path:
        Path to copy installed pip packages to.
    :param bool use_requirements:
        If set, only the packages in the requirements.txt file are installed.
        The requirements.txt file needs to be in the same directory as the
        project which shall be deployed.
        Defaults to false and installs all pacakges found via pip freeze if
        not set.
    :param str local_package:
        The path to a local package with should be included in the deploy as
        well (and/or is not available on PyPi)
    """
    installed_packages = {
        p.project_name: p.as_requirement()
        for p in pip.get_installed_distributions()
    }

    packages = {}
    if not use_requirements:
        print('Gathering pip packages')
        packages.update(installed_packages)
    else:
        if os.path.exists('requirements.txt'):
            print('Gathering requirement packages')
            blacklist = ['-i', '#', 'Python==', 'python-lambda==']
            data = read('requirements.txt')
            for line in data.splitlines():
                if any(line.startswith(black) for black in blacklist): continue
                if line.startswith('-e '):
                    line = line.replace('-e ', '')

                req = packaging.requirements.Requirement(line)
                if len(req.specifier) != 1 or list(req.specifier)[0].operator != '==':
                    # No version specified, use version currently installed:
                    try:
                        req.specifier = installed_packages[req.name].specifier
                    except KeyError:
                        print('WARNING: Unable to resolve version for requirement "{}"; please pip install first.'.format(line))
                        continue
                packages[req.name] = req

    if not packages:
        print('No dependency packages installed!')

    if local_package is not None:
        if not isinstance(local_package, (list, tuple)):
            local_package = [local_package]
        packages.update({
            req.name: req
            for req in [packaging.requirements.Requirement(p) for p in local_package]
        })

    _install_packages(path, packages, cfg.get('runtime', 'python2.7'))


def get_role_name(region, account_id, role):
    """Shortcut to insert the `account_id` and `role` into the iam string."""
    prefix = ARN_PREFIXES.get(region, 'aws')
    return 'arn:{0}:iam::{1}:role/{2}'.format(prefix, account_id, role)


def get_account_id(profile_name, aws_access_key_id, aws_secret_access_key, region=None):
    """Query STS for a users' account_id"""
    client = get_client(
        'sts', profile_name, aws_access_key_id, aws_secret_access_key,
        region,
    )
    return client.get_caller_identity().get('Account')


def get_client(client, profile_name, aws_access_key_id, aws_secret_access_key, region=None):
    """Shortcut for getting an initialized instance of the boto3 client."""

    boto3.setup_default_session(
        profile_name=profile_name,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region,
    )
    return boto3.client(client)


def create_function(cfg, path_to_zip_file, *use_s3, **s3_file):
    """Register and upload a function to AWS Lambda."""

    print('Creating your new Lambda function')
    byte_stream = read(path_to_zip_file, binary_file=True)
    profile_name = cfg.get('profile')
    aws_access_key_id = cfg.get('aws_access_key_id')
    aws_secret_access_key = cfg.get('aws_secret_access_key')

    account_id = get_account_id(
        profile_name, aws_access_key_id, aws_secret_access_key, cfg.get('region'),
    )
    role = get_role_name(
        cfg.get('region'), account_id,
        cfg.get('role', 'lambda_basic_execution'),
    )

    client = get_client(
        'lambda', profile_name, aws_access_key_id, aws_secret_access_key,
        cfg.get('region'),
    )

    # Do we prefer development variable over config?
    buck_name = (
        os.environ.get('S3_BUCKET_NAME') or cfg.get('bucket_name')
    )
    func_name = (
        os.environ.get('LAMBDA_FUNCTION_NAME') or cfg.get('function_name')
    )
    print('Creating lambda function with name: {}'.format(func_name))

    if use_s3:
        kwargs = {
            'FunctionName': func_name,
            'Runtime': cfg.get('runtime', 'python2.7'),
            'Role': role,
            'Handler': cfg.get('handler'),
            'Code': {
                'S3Bucket': '{}'.format(buck_name),
                'S3Key': '{}'.format(s3_file),
            },
            'Description': cfg.get('description'),
            'Timeout': cfg.get('timeout', 15),
            'MemorySize': cfg.get('memory_size', 512),
            'Publish': True,
        }
    else:
        kwargs = {
            'FunctionName': func_name,
            'Runtime': cfg.get('runtime', 'python2.7'),
            'Role': role,
            'Handler': cfg.get('handler'),
            'Code': {'ZipFile': byte_stream},
            'Description': cfg.get('description'),
            'Timeout': cfg.get('timeout', 15),
            'MemorySize': cfg.get('memory_size', 512),
            'Publish': True,
        }

    if 'environment_variables' in cfg:
        kwargs.update(
            Environment={
                'Variables': {
                    key: get_environment_variable_value(value)
                    for key, value
                    in cfg.get('environment_variables').items()
                },
            },
        )

    client.create_function(**kwargs)


def update_function(cfg, path_to_zip_file, *use_s3, **s3_file):
    """Updates the code of an existing Lambda function"""

    print('Updating your Lambda function')
    byte_stream = read(path_to_zip_file, binary_file=True)
    profile_name = cfg.get('profile')
    aws_access_key_id = cfg.get('aws_access_key_id')
    aws_secret_access_key = cfg.get('aws_secret_access_key')

    account_id = get_account_id(
        profile_name, aws_access_key_id, aws_secret_access_key, cfg.get('region'),
    )
    role = get_role_name(
        cfg.get('region'), account_id,
        cfg.get('role', 'lambda_basic_execution'),
    )

    client = get_client(
        'lambda', profile_name, aws_access_key_id, aws_secret_access_key,
        cfg.get('region'),
    )

    # Do we prefer development variable over config?
    buck_name = (
        os.environ.get('S3_BUCKET_NAME') or cfg.get('bucket_name')
    )

    if use_s3:
        client.update_function_code(
            FunctionName=cfg.get('function_name'),
            S3Bucket='{}'.format(buck_name),
            S3Key='{}'.format(s3_file),
            Publish=True,
        )
    else:
        client.update_function_code(
            FunctionName=cfg.get('function_name'),
            ZipFile=byte_stream,
            Publish=True,
        )

    kwargs = {
        'FunctionName': cfg.get('function_name'),
        'Role': role,
        'Runtime': cfg.get('runtime', 'python2.7'),
        'Handler': cfg.get('handler'),
        'Description': cfg.get('description'),
        'Timeout': cfg.get('timeout', 15),
        'MemorySize': cfg.get('memory_size', 512),
        'VpcConfig': {
            'SubnetIds': cfg.get('subnet_ids', []),
            'SecurityGroupIds': cfg.get('security_group_ids', []),
        },
    }

    if 'environment_variables' in cfg:
        kwargs.update(
            Environment={
                'Variables': {
                    key: str(get_environment_variable_value(value))
                    for key, value
                    in cfg.get('environment_variables').items()
                },
            },
        )

    client.update_function_configuration(**kwargs)


def upload_s3(cfg, path_to_zip_file, *use_s3):
    """Upload a function to AWS S3."""

    print('Uploading your new Lambda function')
    profile_name = cfg.get('profile')
    aws_access_key_id = cfg.get('aws_access_key_id')
    aws_secret_access_key = cfg.get('aws_secret_access_key')
    client = get_client(
        's3', profile_name, aws_access_key_id, aws_secret_access_key,
        cfg.get('region'),
    )
    byte_stream = b''
    with open(path_to_zip_file, mode='rb') as fh:
        byte_stream = fh.read()
    s3_key_prefix = cfg.get('s3_key_prefix', '/dist')
    checksum = hashlib.new('md5', byte_stream).hexdigest()
    timestamp = str(time.time())
    filename = '{prefix}{checksum}-{ts}.zip'.format(
        prefix=s3_key_prefix, checksum=checksum, ts=timestamp,
    )

    # Do we prefer development variable over config?
    buck_name = (
        os.environ.get('S3_BUCKET_NAME') or cfg.get('bucket_name')
    )
    func_name = (
        os.environ.get('LAMBDA_FUNCTION_NAME') or cfg.get('function_name')
    )
    kwargs = {
        'Bucket': '{}'.format(buck_name),
        'Key': '{}'.format(filename),
        'Body': byte_stream,
    }

    client.put_object(**kwargs)
    print('Finished uploading {} to S3 bucket {}'.format(func_name, buck_name))
    if use_s3:
        return filename


def function_exists(cfg, function_name):
    """Check whether a function exists or not"""

    profile_name = cfg.get('profile')
    aws_access_key_id = cfg.get('aws_access_key_id')
    aws_secret_access_key = cfg.get('aws_secret_access_key')
    client = get_client(
        'lambda', profile_name, aws_access_key_id, aws_secret_access_key,
        cfg.get('region'),
    )

    # Need to loop through until we get all of the lambda functions returned.
    # It appears to be only returning 50 functions at a time.
    functions = []
    functions_resp = client.list_functions()
    functions.extend([
        f['FunctionName'] for f in functions_resp.get('Functions', [])
    ])
    while('NextMarker' in functions_resp):
        functions_resp = client.list_functions(
            Marker=functions_resp.get('NextMarker'),
        )
        functions.extend([
            f['FunctionName'] for f in functions_resp.get('Functions', [])
        ])
    return function_name in functions


def read_cfg(src, config_file_paths, profile_name):
    if isinstance(config_file_paths, str):
        # Accept singular string, convert to list:
        config_file_paths = config_file_paths.split(':')
    for path in config_file_paths:
        path = os.path.join(src, path)
        if os.path.isfile(path):
            cfg = read(path, loader=yaml.load)
            if profile_name is not None:
                cfg['profile'] = profile_name
            elif 'AWS_PROFILE' in os.environ:
                cfg['profile'] = os.environ['AWS_PROFILE']
            return cfg
    raise FileNotFoundError('Unable to find config file: %s' % ', '.join(config_file_paths))


def manylinux_wheel_file_suffix(python_runtime):
    if python_runtime == 'python2.7':
        return 'cp27mu-manylinux1_x86_64.whl'
    elif python_runtime == 'python3.6':
        return 'cp36m-manylinux1_x86_64.whl'
    else:
        raise ValueError('Unknown Python runtime: {}'.format(python_runtime))


# Borrowed from <https://github.com/Miserlou/Zappa/blame/4d2d40cf7/zappa/core.py#L802-L828>.
# (C) Rich "Miserlou" Jones and respective authors | License: MIT
def get_cached_manylinux_wheel(package_name, package_version, python_runtime, show_progress=True):
    """
    Gets the locally stored version of a manylinux wheel. If one does not exist, the function downloads it.
    """
    cached_wheels_dir = os.path.join(tempfile.gettempdir(), 'cached_wheels')
    if not os.path.isdir(cached_wheels_dir):
        os.makedirs(cached_wheels_dir)

    wheel_file_suffix = manylinux_wheel_file_suffix(python_runtime)
    wheel_file = '{0!s}-{1!s}-{2!s}'.format(package_name, package_version, wheel_file_suffix)
    wheel_path = os.path.join(cached_wheels_dir, wheel_file)

    if not os.path.exists(wheel_path) or not zipfile.is_zipfile(wheel_path):
        # The file is not cached, download it.
        wheel_url = get_manylinux_wheel_url(package_name, package_version, wheel_file_suffix)

        print(" - {}=={}: Downloading".format(package_name, package_version))
        with open(wheel_path, 'wb') as f:
            download_url_to_stream(wheel_url, f, show_progress)

        if not zipfile.is_zipfile(wheel_path):
            raise Exception('Wheel downloaded from PyPI is not a ZIP file: {}'.format(wheel_path))
    else:
        print(" - {}=={}: Using locally cached manylinux wheel".format(package_name, package_version))

    return wheel_path


# Borrowed from <https://github.com/Miserlou/Zappa/blame/4d2d40cf7/zappa/core.py#L830-L873>.
# (C) Rich "Miserlou" Jones and respective authors | License: MIT
def get_manylinux_wheel_url(package_name, package_version, wheel_file_suffix):
    """
    For a given package name, returns a link to the download URL,
    else returns None.

    This function downloads metadata JSON of `package_name` from Pypi
    and examines if the package has a manylinux wheel. This function
    also caches the JSON file so that we don't have to poll Pypi
    every time.
    """
    cached_pypi_info_dir = os.path.join(tempfile.gettempdir(), 'cached_pypi_info')
    if not os.path.isdir(cached_pypi_info_dir):
        os.makedirs(cached_pypi_info_dir)
    # Even though the metadata is for the package, we save it in a
    # filename that includes the package's version. This helps in
    # invalidating the cached file if the user moves to a different
    # version of the package.
    json_file = '{0!s}-{1!s}.json'.format(package_name, package_version)
    json_file_path = os.path.join(cached_pypi_info_dir, json_file)
    if os.path.exists(json_file_path):
        with open(json_file_path, mode='r', encoding='utf-8') as metafile:
            data = json.load(metafile)
    else:
        url = 'https://pypi.python.org/pypi/{}/json'.format(package_name)
        try:
            res = requests.get(url, timeout=1.5)
            data = res.json()
        except Exception as e:
            raise IOError('Unable to download package metadata from {}: {}'.format(url, e))
        with open(json_file_path, mode='w', encoding='utf-8') as metafile:
            jsondata = json.dumps(data)
            metafile.write(jsondata)

    if package_version not in data['releases']:
        raise ValueError('PyPI has no release for package: {}=={}'.format(package_name, package_version))

    for f in data['releases'][package_version]:
        if f['filename'].endswith(wheel_file_suffix):
            return f['url']

    raise ValueError('PyPI has no wheel for package: {}=={} *.{}'.format(
        package_name, package_version, wheel_file_suffix))


# Borrowed from <https://github.com/Miserlou/Zappa/blame/95a8e0860/zappa/core.py#L746-L765>.
# (C) Rich "Miserlou" Jones and respective authors | License: MIT
#def get_installed_packages():
#    """
#    Returns a dict of installed packages that Zappa cares about.
#    """
#    import pip  # this is to avoid 'funkiness' with global import
#
#    python_runtime = 'python{}.{}'.format(sys.version_info.major, sys.version_info.minor)
#    site_packages    = os.path.join(venv, 'lib',   python_runtime, 'site-packages')
#    site_packages_64 = os.path.join(venv, 'lib64', python_runtime, 'site-packages')
#
#    packages_to_keep = []
#    if os.path.isdir(site_packages):
#        packages_to_keep += os.listdir(site_packages)
#    if os.path.isdir(site_packages_64):
#        packages_to_keep += os.listdir(site_packages_64)
#
#    packages_to_keep = [package.lower() for package in packages_to_keep]
#
#    installed_packages = {
#        package.project_name.lower(): package.version
#        for package in pip.get_installed_distributions()
#        if package.project_name.lower() in packages_to_keep or
#           package.location in [site_packages, site_packages_64]
#    }
#
#    return installed_packages
