# Curriculum strategy

The policy can learn holonomous walking from scratch if not bothered too much.
But if you get weird walking on the first run, try reducing the yaw vel command first.
TODO other tips

## First run
- no domain rand
- no push
- plane terrain

Stop when you have a nice holonomous walk.

## Second run
Resume the training on the last checkpoint of the previous run
- Add rough terrain. But not very rough at first, just a little

## Third run
- Add domain randomization

## Fourth run
- Add push
