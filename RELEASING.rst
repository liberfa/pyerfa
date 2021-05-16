Release instructions
====================

For a release based on a new version of ``erfa``, first set the
submodule to the right version, and make sure to clean any leftover
files from earlier compilation::

    cd liberfa/erfa
    git fetch origin
    git checkout v1.7.0
    git clean -fxd
    cd ../..

For any release, first do a last check that things are OK in a clean
environment::

    git clean -fxd
    tox -e test

It is also wise to check that the new release works with astropy.
We can use the ``tox`` environment we just created, but have to be sure
to leave the package directory::

    source .tox/test/bin/activate
    cd ~  #
    pip install astropy[test]
    pytest --pyargs astropy --remote-data=any
    cd -  # Back to the package directory.
    deactivate
    git clean -fxd

Once the package is ready to release, first edit ``CHANGES.rst`` to
add the release data (and add any information needed, such as an
update to the bundled ``liberfa``).

Then, use ``git tag`` to tag the release::

    git tag -m <version> <version>

e.g::

    git tag -m v1.7.0 v1.7.0

Here, the version number should generally be the same as that of
the erfa library that is included (and you should make sure the
git submodule ``liberfa/erfa`` is at the correct tag as well).
If there is a need for just the wrappers to be updated, add a
fourth number to the release (e.g., ``1.7.0.1``).

You can also include the ``-s`` flag to sign the tag if you have
PGP keys set up. Then, push the tag to GitHub, e.g.::

    git push upstream v0.1

and the build should happen automatically on Azure pipelines. You can
follow the progress of the build here:

https://dev.azure.com/liberfa/pyerfa/_build

If there are any failures, you can always delete the tag, fix the
issues, tag the release again, and push the tag to GitHub.

See the `OpenAstronomy Azure Pipelines Templates Documentation <https://openastronomy-azure-pipelines.readthedocs.io/en/latest/publish.html>`_
for more details about the Azure Pipelines set-up.
