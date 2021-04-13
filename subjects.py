# subjects.py: My current easiest solution to save subject ids without fancier 
# text reading methods. Might change this soon as part of general config file!

# TODO: All config info should be replace with a more automatic config file creation method in the near future
# (it would help with getting behavioral data from subject ids too.)
# excluded binge subs: 225, 2604, 5497
subs_binge = [91, 283, 595, 413, 579, 685, 1025, 630, 725, 1237, 1046, 1132, 300, 889,
            1352, 2800, 1976, 2689, 1211, 1961, 65, 4592, 2199, 4829, 2839, 3010, 3002, 2834, 3225,
            3324, 3525, 4930, 3464, 5293, 3364, 4022, 4081, 4129, 5064, 4591, 5387, 5469, 5579, 
            6638, 6018, 5906, 7124]

# excluded smoke subs: 1438
subs_smoke = [31, 192, 232, 211, 275, 233, 472, 96, 443, 538, 811, 623, 479, 1105, 667, 772, 1646,
            698, 615, 1517, 1525, 1190, 731, 885, 2232, 897, 902, 857, 931, 1021, 1108, 1376, 1189,
            1143, 1399, 2143, 2133, 1860, 1481, 1585, 1636, 1652, 1640, 2001, 2099, 2577, 2130,
            2265, 2696]

subjects = subs_binge + subs_smoke