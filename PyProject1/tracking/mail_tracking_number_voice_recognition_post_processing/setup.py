from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='mail_tracking_number_voice_recognition_post_processing',
      version='0.0.1',
      description='Library contains functionality for converting given nn output into tracking number (Russian Post or '
                  'International mail) format',
      long_description=long_description,
      long_description_content_type="text/markdown",
      packages=find_packages(),
      include_package_data=True,
      author_email='nik310392illbeback@gmail.com',
      zip_safe=False)
