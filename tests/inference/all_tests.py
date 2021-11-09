import pathlib
import sys

# Get the package directory
package_dir = str(pathlib.Path(__file__).resolve().parents[3])
# Add the package directory into sys.path if necessary
if package_dir not in sys.path:
    sys.path.insert(0, package_dir)

import pyb4ml.tests.inference.be_misconception_test
import pyb4ml.tests.inference.be_student_test
import pyb4ml.tests.inference.bp_student_test
import pyb4ml.tests.inference.go_extended_student_test
