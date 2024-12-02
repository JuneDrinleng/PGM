import math
import numpy as np
from matplotlib import pyplot as plt
# edge weight
w_11=w_12=w_13=-1
w_21=w_22=w_23=0.2
# w_11=w_12=w_13=-1    w_21=w_22=w_23=0.2

# node H weight
alpha_1=0.1
alpha_2=0.1

# node V weight
beta_1=0.5
beta_2=-0.3
beta_3=-0.5

# define clique
# C_1 is {H_1,V_1}, C_2 is {H_1,V_2}, C_3 is {H_1,V_3}, C_4 is {H_2,V_1}, C_5 is {H_2,V_2}, C_6 is {H_2,V_3}

# origin distribution
origin_belief=[0.5]*5 # order:0: H_1,1: H_2,2: V_1,3: V_2,4: V_3

origin_p_H1_1=origin_belief[0] # p(H1=1)
origin_p_H2_1=origin_belief[1] # p(H2=1)
origin_p_V1_1=origin_belief[2] # p(V1=1)
origin_p_V2_1=origin_belief[3] # p(V2=1)
origin_p_V3_1=origin_belief[4] # p(V3=1)

origin_p_H1_0=1-origin_p_H1_1 # p(H1=0)
origin_p_H2_0=1-origin_p_H2_1 # p(H2=0)
origin_p_V1_0=1-origin_p_V1_1 # p(V1=0)
origin_p_V2_0=1-origin_p_V2_1 # p(V2=0)
origin_p_V3_0=1-origin_p_V3_1 # p(V3=0)

message_12=np.zeros(2)
message_23=np.zeros(2)
message_34=np.zeros(2)
message_45=np.zeros(2)
message_56=np.zeros(2)
message_65=np.zeros(2)
message_54=np.zeros(2)
message_43=np.zeros(2)
message_32=np.zeros(2)
message_21=np.zeros(2)
miu_12=np.array([1,1])
miu_23=np.array([1,1])
miu_34=np.array([1,1])
miu_45=np.array([1,1])
miu_56=np.array([1,1])
miu_65=np.array([1,1])
miu_54=np.array([1,1])
miu_43=np.array([1,1])
miu_32=np.array([1,1])
miu_21=np.array([1,1])
history_pH1=[]
history_pH2=[]
history_pV1=[]
history_pV2=[]
history_pV3=[]
history_pH1.append(origin_p_H1_0)
history_pH2.append(origin_p_H2_0)
history_pV1.append(origin_p_V1_0)
history_pV2.append(origin_p_V2_0)
history_pV3.append(origin_p_V3_0)
for iter in range(0,100):
    for H1 in [0,1]:
        message_1=origin_p_V1_1*math.exp(alpha_1*H1+beta_1+w_11*H1)+origin_p_V1_0*math.exp(alpha_1*H1) 
        message_12[H1]=message_1
        message_2=message_1*(origin_p_V2_1*math.exp(alpha_1*H1+beta_2+w_12*H1)+origin_p_V2_0*math.exp(alpha_1*H1))
        message_23[H1]=message_2
    # message_12=[message_12(H1=0),message_12(H1=1)]
    # message_23=[message_23(H1=0),message_23(H1=1)]
    message_12=np.array(message_12)/miu_12
    message_23=np.array(message_23)/miu_23
    # message_12=np.array(message_12)/np.sum(message_12)
    # message_23=np.array(message_23)/np.sum(message_23)
    phi_1=message_12
    phi_2=message_23


    for V3 in [0,1]:
        message_3=message_23[1]*origin_p_H1_1*math.exp(alpha_1+beta_3*V3+w_13*V3)+origin_p_H1_0*math.exp(beta_3*V3)*message_23[0]
        message_34[V3]=message_3
    # message_34=[message_34(V3=0),message_34(V3=1)]
    message_34=np.array(message_34)/miu_34
    # message_34=np.array(message_34)/np.sum(message_34)
    phi_3=message_34


    for H2 in [0,1]:
        message_4=origin_p_V3_1*message_34[1]*math.exp(alpha_2*H2+beta_3+w_23*H2)+origin_p_V3_0*message_34[0]*math.exp(alpha_2*H2)
        message_45[H2]=message_4
        message_5=origin_p_V2_1*message_4*math.exp(alpha_2*H2+beta_2+w_22*H2)+origin_p_V2_0*message_4*math.exp(alpha_2*H2)
        message_56[H2]=message_5
        message_6=message_5*(origin_p_V1_1*math.exp(alpha_2*H2+beta_1+w_21*H2)+origin_p_V1_0*math.exp(alpha_2*H2))
        message_65[H2]=message_6
        message_7=message_6*(origin_p_V2_1*math.exp(alpha_2*H2+beta_2+w_22*H2)+origin_p_V2_0*math.exp(alpha_2*H2))
        message_54[H2]=message_7
    # message_45=[message_45(H2=0),message_45(H2=1)]
    # message_56=[message_56(H2=0),message_56(H2=1)]
    # message_65=[message_65(H2=0),message_65(H2=1)]
    # message_54=[message_54(H2=0),message_54(H2=1)]
    message_45=np.array(message_45)/miu_45
    message_56=np.array(message_56)/miu_56
    message_65=np.array(message_65)/miu_65
    message_54=np.array(message_54)/miu_54
    # message_45=np.array(message_45)/np.sum(message_45)
    # message_56=np.array(message_56)/np.sum(message_56)
    # message_65=np.array(message_65)/np.sum(message_65)
    # message_54=np.array(message_54)/np.sum(message_54)
    phi_4=message_45
    phi_5=message_56
    phi_6=message_65
    phi_5=message_54




    for V3 in [0,1]:
        message_8=message_54[1]*origin_p_H2_1*math.exp(alpha_2+beta_3*V3+w_23*V3)+origin_p_H2_0*math.exp(beta_3*V3)*message_54[0]
        message_43[V3]=message_8
    # message_43=[message_43(V3=0),message_43(V3=1)]
    message_43=np.array(message_43)/miu_43
    # message_43=np.array(message_43)/np.sum(message_43)
    phi_3=message_43


    for H1 in [0,1]:
        message_9=origin_p_V3_1*message_43[1]*math.exp(alpha_1*H1+beta_3+w_13*H1)+origin_p_V3_0*message_43[0]*math.exp(alpha_1*H1)
        message_32[H1]=message_9
        message_10=message_9*(origin_p_V2_1*math.exp(alpha_1*H1+beta_2+w_12*H1)+origin_p_V2_0*math.exp(alpha_1*H1))
        message_21[H1]=message_10
    # message_32=[message_32(H1=0),message_32(H1=1)]
    # message_21=[message_21(H1=0),message_21(H1=1)]
    message_32=np.array(message_32)/miu_32
    message_21=np.array(message_21)/miu_21
    # message_32=np.array(message_32)/np.sum(message_32)
    # message_21=np.array(message_21)/np.sum(message_21)
    phi_2=message_32
    phi_1=message_21

    origin_p_V1_0=phi_1[0]*math.exp(0)+phi_1[1]*math.exp(alpha_1)
    origin_p_V1_1=phi_1[0]*math.exp(beta_1)+phi_1[1]*math.exp(alpha_1+beta_1+w_11)
    origin_p_V1_0=origin_p_V1_0/(origin_p_V1_0+origin_p_V1_1)
    origin_p_V1_1=origin_p_V1_1/(origin_p_V1_0+origin_p_V1_1)

    origin_p_V2_0=phi_2[0]*math.exp(0)+phi_2[1]*math.exp(alpha_1)
    origin_p_V2_1=phi_2[0]*math.exp(beta_2)+phi_2[1]*math.exp(alpha_1+beta_2+w_12)
    origin_p_V2_0=origin_p_V2_0/(origin_p_V2_0+origin_p_V2_1)
    origin_p_V2_1=origin_p_V2_1/(origin_p_V2_0+origin_p_V2_1)

    origin_p_V3_0=phi_3[0]*math.exp(0)+phi_3[1]*math.exp(alpha_1)
    origin_p_V3_1=phi_3[0]*math.exp(beta_3)+phi_3[1]*math.exp(alpha_1+beta_3+w_13)
    origin_p_V3_0=origin_p_V3_0/(origin_p_V3_0+origin_p_V3_1)
    origin_p_V3_1=origin_p_V3_1/(origin_p_V3_0+origin_p_V3_1)

    origin_p_H1_0=phi_1[0]*(math.exp(0)+math.exp(beta_1))
    origin_p_H1_1=phi_1[1]*(math.exp(alpha_1)+math.exp(alpha_1+beta_1+w_11))
    origin_p_H1_0=origin_p_H1_0/(origin_p_H1_0+origin_p_H1_1)
    origin_p_H1_1=origin_p_H1_1/(origin_p_H1_0+origin_p_H1_1)

    origin_p_H2_0=phi_5[0]*(math.exp(0)+math.exp(beta_2))
    origin_p_H2_1=phi_5[1]*(math.exp(alpha_2)+math.exp(alpha_2+beta_2+w_22))
    origin_p_H2_0=origin_p_H2_0/(origin_p_H2_0+origin_p_H2_1)
    origin_p_H2_1=origin_p_H2_1/(origin_p_H2_0+origin_p_H2_1)

    miu_12=message_12
    miu_23=message_23
    miu_34=message_34
    miu_45=message_45
    miu_56=message_56
    miu_65=message_65
    miu_54=message_54
    miu_43=message_43
    miu_32=message_32
    miu_21=message_21
    if math.fabs(origin_p_H1_0-history_pH1[-1])>1e-6:
        history_pH1.append(origin_p_H1_0)
        history_pH2.append(origin_p_H2_0)
        history_pV1.append(origin_p_V1_0)
        history_pV2.append(origin_p_V2_0)
        history_pV3.append(origin_p_V3_0)
        stable_iter=iter+2
    else:
        history_pH1.append(origin_p_H1_0)
        history_pH2.append(origin_p_H2_0)
        history_pV1.append(origin_p_V1_0)
        history_pV2.append(origin_p_V2_0)
        history_pV3.append(origin_p_V3_0)
    
    pass
plt.plot(range(len(history_pH1)),history_pH1,marker='o',label='p(H1=0)')
plt.plot(range(len(history_pH2)),history_pH2,marker='o',label='p(H2=0)')
plt.plot(range(len(history_pV1)),history_pV1,marker='o',label='p(V1=0)')
plt.plot(range(len(history_pV2)),history_pV2,marker='o',label='p(V2=0)')
plt.plot(range(len(history_pV3)),history_pV3,marker='o',label='p(V3=0)')
plt.xlabel('iteration_number')
plt.ylabel('probability')
plt.legend()
plt.ylim(0,1)
plt.savefig('T2.png')
print(f'H1:{history_pH1[-1]}\n')
print(f'H2:{history_pH2[-1]}\n')
print(f'V1:{history_pV1[-1]}\n')
print(f'V2:{history_pV2[-1]}\n')
print(f'V3:{history_pV3[-1]}\n')
print('iteration_number:',iter)