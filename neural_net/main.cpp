#include <iostream>
#include "dataloader.h"
#include "network.h"
#include "pmad.h"
#include "train.h"

int main() {
    const std::string data_dir = "data/Emnist Letters";

    std::cout << "Loading datasets...\n";
    ImageDataset train_set = load_emnist_letters(data_dir, /*train=*/true,  /*do_normalize=*/false);
    ImageDataset test_set  = load_emnist_letters(data_dir, /*train=*/false);
    std::printf("  train: %zu images\n", train_set.images.size());
    std::printf("  test : %zu images\n\n", test_set.images.size());

    visualize(train_set.images[0], train_set.labels[0], 'A' + train_set.labels[0]);
    std::cout << "\n";
    normalize(train_set.images);

    // PMAD must be initialised before Network — Layer constructors alloc from it
    init_network_pmad();

    {
        Network net;

        print_pmad_plan(net);
        print_pmad_stats();

        train_batched(net, train_set, test_set);
    }  // ~Network: Layer destructors return all 12 vectors to PMAD free lists

    destroy_network_pmad();
    return 0;
}
