#include <iostream>
#include "dataloader.h"
#include "network.h"
#include "train.h"

int main() {
    const std::string data_dir = "data/Emnist Letters";

    std::cout << "Loading datasets...\n";
    ImageDataset train_set = load_emnist_letters(data_dir, /*train=*/true,  /*do_normalize=*/false);
    ImageDataset test_set  = load_emnist_letters(data_dir, /*train=*/false);
    std::printf("  train: %zu images\n", train_set.images.size());
    std::printf("  test : %zu images\n\n", test_set.images.size());

    // Visualize first raw sample, then normalize in-place
    visualize(train_set.images[0], train_set.labels[0], 'A' + train_set.labels[0]);
    std::cout << "\n";
    normalize(train_set.images);

    Network net;
    train(net, train_set, test_set);

    return 0;
}
