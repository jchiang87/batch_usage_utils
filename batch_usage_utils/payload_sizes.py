__all__ = ["payload_sizes"]

payload_sizes = {
    # stage 1
    "isr": 10,
    "calibrateImage": 10,
    "standardizeSingleVisitStar": 10,
    "consolidateVisitSummary": 10,
    "consolidateSingleVisitStar": 10,
    # stage 2
    "refitPsfModelDetector": 10,
    "consolidateRefitPsfModelDetector": 10,
    "recalibrateSingleVisitStar": 10,
    "standardizeRecalibratedStar": 10,
    "consolidateRecalibratedStar": 10,
    # stage 3
    "makeDirectWarp": 10,
    "makePsfMatchedWarp": 10,
    "selectDeepCoaddVisits": 10,
    "selecTemplateCoaddVisits": 10,
    # stage 4
    "reprocessVisitImage": 10,
    "standardizeSource": 10,
    "rewarpTemplate": 10,
    "subtractImages": 10,
    "detectAndMeasureDiaSource": 10,
    "filterDiaSource": 10,
    "forcedPhotObjectDetector": 10,
    "computeReliability": 10,
    "filterDiaSourceReliability": 10,
    "standardizeDiaSource": 10,
    "forcedPhotDiaObjectDetector": 10,
}
