#include <cstdio>
#include "dataloader.h"
#include "network.h"
#include "pmad.h"
#include "train.h"

int main() {
    const std::string data_dir = "data/Emnist Letters";

    std::printf("Loading datasets...\n");
    ImageDataset train_set = load_emnist_letters(data_dir, true,  false);
    ImageDataset test_set  = load_emnist_letters(data_dir, false);
    std::printf("  train: %d images (%.0f MB)\n",
                train_set.cols(), train_set.images.size() * sizeof(float) / 1048576.0);
    std::printf("  test : %d images\n\n", test_set.cols());

    visualize(train_set.images.col(0), train_set.labels[0],
              static_cast<char>('A' + train_set.labels[0]));
    std::printf("\n");
    normalize(train_set.images);

    init_network_pmad();
    {
        Network net;
        print_pmad_plan(net);
        print_pmad_stats();
        train_batched(net, train_set, test_set);
    }
    destroy_network_pmad();
    return 0;
}
