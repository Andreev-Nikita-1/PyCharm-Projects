import numpy as np
import matplotlib.pyplot as plt
import itertools

# v17 = list(itertools.accumulate([7.92, 36.16, 25.09, 27.49, 3.34]))
# v18 = list(itertools.accumulate([3.85, 31.13, 32.96, 29.92, 2.13]))
#
# low17 = [0, 0, v17[0], v17[0], v17[1], v17[1], v17[2], v17[2], v17[3], v17[3], v17[4]]
# high17 = [v17[0], v17[0], v17[1], v17[1], v17[2], v17[2], v17[3], v17[3], v17[4], v17[4], v17[4]]
#
# low18 = [0, 0, v18[0], v18[0], v18[1], v18[1], v18[2], v18[2], v18[3], v18[3], v18[4]]
# high18 = [v18[0], v18[0], v18[1], v18[1], v18[2], v18[2], v18[3], v18[3], v18[4], v18[4], v18[4]]
#
# xs = [0, 20, 20, 40, 40, 60, 60, 80, 80, 100, 100]
#
# plt.plot(xs, low17, 'r')
# plt.plot(xs, high17, 'r')
# plt.plot(xs, low18, 'g')
# plt.plot(xs, high18, 'g')
# plt.show()
#
# sup = max([max(high17[i] - low18[i], high18[i] - low17[i]) for i in range(len(xs))]) / 100
# print(np.sqrt(3.9 * 100000 / 2) * sup)

s = """29. The Monitoring Trustee shall:
(a) propose in its first report to the Commission a detailed work plan describing how it
intends to monitor compliance with the obligations and conditions attached to the
Decision.
(b) oversee, in close co-operation with the Hold Separate Manager, the on-going
management of the CEE Divestment Business with a view to ensuring its continued
economic viability, marketability and competitiveness and monitor compliance by the
Notifying Party with the conditions and obligations attached to the Decision. To that
end the Monitoring Trustee shall:
(i) monitor the preservation of the economic viability, marketability and
competitiveness of the CEE Divestment Business, and the keeping separate
of the CEE Divestment Business from the business retained by the Parties, in
accordance with paragraphs 11 and 12 of these CEE Commitments;
(ii) supervise the management of the CEE Divestment Business as a distinct and
saleable entity, in accordance with paragraph 13 of these CEE Commitments;
(iii)with respect to Confidential Information:
 determine all necessary measures to ensure that the Notifying Party
does not after Completion obtain any Confidential Information relating
to the CEE Divestment Business,
 in particular strive for the severing of the CEE Divestment Business’
participation in a central information technology network to the extent
possible, without compromising the viability of the CEE Divestment
Business,
 make sure that any Confidential Information relating to the CEE
Divestment Business obtained by the Notifying Party before
Completion is properly ring-fenced and will not be used by the
Notifying Party, and
 decide whether such information may be disclosed to or kept by the
Notifying Party as the disclosure is reasonably necessary to allow the
Notifying Party to carry out the divestiture or as the disclosure is
required by law;
(iv) monitor the splitting of assets and the allocation of Personnel between the
CEE Divestment Business and the Notifying Party or Affiliated
Undertakings;
11
(c) propose to the Notifying Party such measures as the Monitoring Trustee considers
necessary to ensure the Notifying Party’s compliance with the conditions and
obligations attached to the Decision, in particular the maintenance of the full
economic viability, marketability or competitiveness of the CEE Divestment Business,
the holding separate of the CEE Divestment Business and the non-disclosure of
competitively sensitive information;
(d) review and assess potential purchasers as well as the progress of the divestiture
process and verify that, dependent on the stage of the divestiture process:
(i) potential purchasers receive sufficient and correct information relating to the
CEE Divestment Business and the Personnel in particular by reviewing, if
available, the data room documentation, the information memorandum and
the due diligence process, and
(ii) potential purchasers are granted reasonable access to the Personnel;
(e) act as a contact point for any requests by third parties, in particular potential
purchasers, in relation to the CEE Commitments;
(f) provide to the Commission, sending the Notifying Party a non-confidential copy at the
same time, a written report within 15 days after the end of every month that shall
cover the operation and management of the CEE Divestment Business as well as the
splitting of assets and the allocation of Personnel so that the Commission can assess
whether the business is held in a manner consistent with the CEE Commitments and
the progress of the divestiture process as well as potential purchasers;
(g) promptly report in writing to the Commission, sending the Notifying Party a nonconfidential copy at the same time, if it concludes on reasonable grounds that the
Notifying Party is failing to comply with these CEE Commitments;
(h) within one week after receipt of the documented proposal referred to in paragraph 19
of these CEE Commitments, submit to the Commission, sending the Notifying Party a
non-confidential copy at the same time, a reasoned opinion as to the suitability and
independence of the proposed purchaser(s) and the viability of the CEE Divestment
Business after the sale and as to whether the CEE Divestment Business is sold in a
manner consistent with the conditions and obligations attached to the Decision, in
particular, if relevant, whether the sale of the CEE Divestment Business without one
or more Assets or not all of the Personnel affects the viability of the CEE Divestment
Business after the sale, taking account the capabilities of the proposed purchaser(s);
(i) assume the other functions assigned to the Monitoring Trustee under the conditions
and obligations attached to the Decision.
30. If the Monitoring and Divestiture Trustee are not the same legal or natural persons, the Monitoring
Trustee and the Divestiture Trustee shall cooperate closely with each other during and for the
purpose of the preparation of the Trustee Divestiture Period in order to facilitate each other's tasks.
Duties and obligations of the Divestiture Trustee
31. Within the Trustee Divestiture Period, the Divestiture Trustee shall sell at no minimum price the
CEE Divestment Business to no more than two purchasers (subject to the proviso that if the CEE
Divestment Business is acquired by two CEE Purchasers, SABMiller Poland and SABMiller 
12
Czech Republic/Slovakia will not be acquired by the same CEE Purchaser), provided that the
Commission has approved both the purchaser and the final binding sale and purchase agreement
(and ancillary agreements) as in line with the Commission's Decision and the CEE Commitments
in accordance with paragraphs 18 and 19 of these CEE Commitments. The Divestiture Trustee
shall include in the sale and purchase agreement (as well as in any ancillary agreements) such
terms and conditions as it considers appropriate for an expedient sale in the Trustee Divestiture
Period. In particular, the Divestiture Trustee may include in the sale and purchase agreement such
customary representations and warranties and indemnities as are reasonably required to effect the
sale. The Divestiture Trustee shall protect the legitimate financial and IP-related interests of the
Notifying Party, subject to the Notifying Party’s unconditional obligation to divest at no minimum
price in the Trustee Divestiture Period.
32. In the Trustee Divestiture Period (or otherwise at the Commission’s request), the Divestiture
Trustee shall provide the Commission with a comprehensive monthly report written in English on
the progress of the divestiture process. Such reports shall be submitted within 15 days after the end
of every month with a simultaneous copy to the Monitoring Trustee and a non-confidential copy to
the Notifying Party."""

print(" ".join(s.split('\n')))
