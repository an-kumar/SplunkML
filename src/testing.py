from gda import SplunkGaussianDiscriminantAnalysis

vote_features = ['handicapped_infants', 'water_project_cost_sharing', 'adoption_of_the_budget_resolution','physician_fee_freeze', 'el_salvador_aid', 'religious_groups_in_schools', 'anti_satellite_test_ban','aid_to_nicaraguan_contras','mx_missile','immigration','synfuels_corporation_cutback','education_spending','superfund_right_to_sue','crime','duty_free_exports']
vote_search = 'source="/Users/ankitkumar/Documents/coding/205Consulting/OpenSource/SplunkML/naivebayes/splunk_votes_correct.txt"'
vote_class = 'party'


reaction_features = ['field%s' % i for i in range(1,45)]
reaction_search = 'source="/Users/ankitkumar/Documents/coding/SplunkML/splunk_second_cont.txt"'
reaction_class = 'success'

snb = SplunkGaussianDiscriminantAnalysis(host="localhost", port=8089, username='admin', password='flower00')
snb.train(reaction_search, reaction_features, reaction_class)
snb.predict(reaction_search, reaction_features, reaction_class)