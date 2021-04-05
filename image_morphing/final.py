from source import GUI, morphing, TriAngle

if __name__ == '__main__':
    fp1 = './graph/' + input('graph1: ')
    fp2 = './graph/' + input('graph2: ')
    GUI.GUI_main(fp1, fp2)
    TriAngle.TriAngle_main()
    morphing.Morph_main(fp1, fp2, 8, 0.01)