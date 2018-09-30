
--[[
  the interface between the "Task" and the "Logger" class.
  the task represents a single learning task/objective,
  with its own data, objective function and specific logging mechanism

  the "logger" class is responsible for logging training performance,
  test results and information concerning the updates

  the "Task" does not depend on any functionality exposed by the logger;
  rather, it exposes to the logger the information that might be used
  to create the contents of log files
--]]

NetworkTraining.LoggerInterface={
  addToLog={
    desc="called after a forward-backward pass; intended for accumulating "..
         "prediction results on the training set, before they get logged "..
         "in a call to \"logEpochEnd\"\narguments:\n"..
         "  output   predictor output for the current batch - passing it "..
         "to the logger enables calculating prediction accuracy "..
         "and output statistics\n"..
         "  target   the ground truth predictions for the current batch\n"..
         "  input    the batch of input data; in case you want to compute "..
         "some fancy statistics on input-output pairs"..
         "  cost     the value of the objective for the current batch\n"..
         "  dfdx     the gradient of the objective for the current batch "..
         "(the error signal)\n"
  },
  logEpochEnd={
    desc="called after a training epoch has been completed; "..
         "intended for computing final accuracies and writing to logs\n"..
         "arguments\n"..
         "  net the network\n"
  },
  logUpdate={
    desc="a function for logging weight update-related information;"..
         "it is called after each weight update;\n"..
         "arguments\n"..
         "net       the network; can be used to extract any network params\n"..
         "optimState the current state of the optimization engine "..
                     "see for example \"optim.sgd\" from the optim package\n"..
         "params    the network parameters after update\n"..
         "gradParams the gradient used to update the parameters\n"
  }
}
