"""Compatibility for opaque Extra Bytes fields in LAS/LAZ headers."""


def install_opaque_extra_bytes_fix():
    """For data type 0, options is a byte count, not metadata flags.

    Some laspy versions interpret that count as scale/offset/no-data flags.
    A nine-byte field then gets three scales and cannot even be opened.
    Only the property getters for opaque fields change; typed dimensions,
    point layouts, VLR bytes and source files stay intact.
    """
    from laspy.vlrs.known import ExtraBytesStruct

    if getattr(ExtraBytesStruct, '_naksha_opaque_fixed', False):
        return

    def opaque_safe(prop):
        def read(self):
            return None if self.data_type == 0 else prop.fget(self)
        return property(read, prop.fset, prop.fdel, prop.__doc__)

    for name in ('scale', 'offset', 'no_data'):
        setattr(ExtraBytesStruct, name, opaque_safe(getattr(ExtraBytesStruct, name)))
    ExtraBytesStruct._naksha_opaque_fixed = True
