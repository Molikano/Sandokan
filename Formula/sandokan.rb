class Sandokan < Formula
  desc "High-performance neural network and deep learning library for C++"
  homepage "https://github.com/Molikano/Sandokan"
  url "https://github.com/Molikano/Sandokan/archive/refs/tags/v0.1.0.tar.gz"
  sha256 "REPLACE_WITH_SHA256_AFTER_TAGGING"
  license "MIT"
  head "https://github.com/Molikano/Sandokan.git", branch: "main"

  depends_on "cmake" => :build
  depends_on "eigen"
  depends_on :macos

  def install
    system "cmake", "-S", ".", "-B", "build",
      "-DSANDOKAN_BUILD_EXAMPLES=OFF",
      *std_cmake_args
    system "cmake", "--build", "build"
    system "cmake", "--install", "build"
  end

  test do
    (testpath/"test.cpp").write <<~CPP
      #include <sandokan.h>
      int main() { return 0; }
    CPP
    system ENV.cxx, "-std=c++17", "test.cpp",
      "-I#{include}",
      "-L#{lib}", "-lpmad",
      "-framework", "Accelerate",
      "-o", "test"
  end
end
