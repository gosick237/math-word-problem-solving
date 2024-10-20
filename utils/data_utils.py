def show_dataset_info(dataset,title):
    print("#####", title, "#####")
    print("  [Data Info]")
    print("* Features: ", dataset.features)
    print("* Num of data: ",dataset.__len__())
    print("* Sample: ", dataset[0])
    print("*"*70, "\n")