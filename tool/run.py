from change_format import ChangeFormat as CF
#
chain_test = CF()
chain_test.from_xml(filename="chain_casl.xml")
chain_test.export_to_xml(filename="chain_casl_old.xml")



