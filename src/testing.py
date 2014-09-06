from gda import SplunkGaussianDiscriminantAnalysis

vote_features = ['handicapped_infants', 'water_project_cost_sharing', 'adoption_of_the_budget_resolution','physician_fee_freeze', 'el_salvador_aid', 'religious_groups_in_schools', 'anti_satellite_test_ban','aid_to_nicaraguan_contras','mx_missile','immigration','synfuels_corporation_cutback','education_spending','superfund_right_to_sue','crime','duty_free_exports']
vote_search = 'source="/Users/ankitkumar/Documents/coding/205Consulting/OpenSource/SplunkML/naivebayes/splunk_votes_correct.txt"'
vote_class = 'party'


reaction_features = ['field%s' % i for i in range(1,45)]
reaction_search = 'source="/Users/ankitkumar/Documents/coding/SplunkML/splunk_second_cont.txt"'
reaction_class = 'success'

username = raw_input("What is your username? ")
password = raw_input("What is your password? ")

# GDA#
gda = SplunkGaussianDiscriminantAnalysis(host="localhost", port=8089, username=username, password=password)
gda.test_accuracy_splunk_search(reaction_search, reaction_search, reaction_features, reaction_class)

#GNB # 
gnb = SplunkGaussianNaiveBayes(host="localhost", port=8089, username=username, password=password)
gnb.test_accuracy_splunk_search(reaction_search, reaction_search, reaction_features, reaction_class)

# LINREG#
slr = SplunkLinearRegression(host="localhost", port=8089, username=username, password=password)
slr.test_accuracy_splunk_search(reaction_search, reaction_search,reaction_features, reaction_class)

