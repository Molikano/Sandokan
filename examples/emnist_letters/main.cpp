#include <cstdio>
#include <sandokan/allocator.h>
#include <sandokan/dataset.h>
#include <sandokan/network.h>
#include <sandokan/train.h>

int main() {
    const std::string      data_dir   = "data/Emnist Letters";
    const std::vector<int> arch       = { 784, 256, 512, 256, 26 };
    const int              batch_size = 128;

    std::printf("Loading datasets...\n");
    ImageDataset train_set = load_emnist_letters(data_dir, true,  false);
    ImageDataset test_set  = load_emnist_letters(data_dir, false);
    std::printf("  train: %d images (%.0f MB raw, mmap-resident)\n",
                train_set.cols(),
                double(train_set.cols()) * train_set.image_size() / 1048576.0);
    std::printf("  test : %d images\n\n", test_set.cols());

    Eigen::VectorXf sample(train_set.image_size());
    train_set.get_image_col(0, sample);
    const int first_label = train_set.label(0);
    visualize(sample, first_label, static_cast<char>('A' + first_label));
    std::printf("\n");

    train_set.compute_normalization();

    init_pmad(arch, batch_size);
    {
        Network net(arch);
        print_pmad_plan(net);
        print_pmad_stats();
        train_batched(net, train_set, test_set);
    }
    destroy_pmad();
    return 0;
}
