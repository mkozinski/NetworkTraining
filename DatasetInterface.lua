
--[[
  the interface between the "DataProvider" and the "Dataset" class.
  the "Dataset" is responsible for reading the data (from mem, disc, etc)
  and performing per-data item data augmentation

--]]

NetworkTraining.DatasetInterface={
  noItems={
    desc="returns the number of data items in the dataset\n"
  },
  item={
    desc="returns the input datum "..
         "and (optionally) the corresponding ground truth annotation, "..
         "associated to the index passed as an argument; "..
         "(a comma separated list is returned);\n"..
         "argument:\n  the index of the requested data\n"
  },
  inputName={
    desc="returns the name of the input indexed by the argument; "..
         "the name is only used for error reporting "..
         "and should enable identifying the problematic data item; "..
         "in case the data is stored in files, it should be the file name; "..
         "for data stored in a big tensor in memory, it could the data "..
         "index in form of a string;\nargument\n  index of the data item\n";
  },
  labelName={
    desc="returns the name of the training ground truth annotation "..
         "indexed by the argument; "..
         "the name is only used for error reporting "..
         "and should enable identifying the problematic data item; "..
         "in case the data is stored in files, it should be the file name; "..
         "for data stored in a big tensor in memory, it could the data "..
         "index in form of a string;\nargument\n  index of the data item\n";
  }
}
