
NetworkTraining= NetworkTraining or {}

require('NetworkTraining.Interfaces')
require('NetworkTraining.TaskInterface')
require('NetworkTraining.LoggerInterface')
require('NetworkTraining.DataProviderInterface')
require('NetworkTraining.DatasetInterface')
require('NetworkTraining.TesterInterface')
require('NetworkTraining.utils')
require('NetworkTraining.Setup')
require('NetworkTraining.LoggerBasic')
require('NetworkTraining.LoggerClassification')
require('NetworkTraining.LoggerComposit')
require('NetworkTraining.DatasetMemory')
require('NetworkTraining.DatasetDisk')
require('NetworkTraining.DataProviderDataset')
require('NetworkTraining.DataProviderGeneric')
require('NetworkTraining.Tester')
require('NetworkTraining.TesterComposit')
require('NetworkTraining.Task')
require('NetworkTraining.LoggerF1')
require('NetworkTraining.DatasetPieces')


return NetworkTraining
