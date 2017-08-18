//
//  DetailViewController.swift
//  OpenCV Demo
//
//  Created by Drew Nibeck on 7/31/17.
//  Copyright © 2017 loc. All rights reserved.
//

import UIKit

class DetailViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    @IBOutlet weak var imageView: UIImageView!
    @IBOutlet weak var areaLabel: UILabel!
    @IBOutlet weak var executionTimeLabel: UILabel!
    @IBOutlet weak var foundItButton: UIButton!
    @IBOutlet weak var reactionButton: UIButton!
    @IBOutlet weak var keylinesToggleLabel: UILabel!
    @IBOutlet weak var keylinesToggle: UISwitch!
    
    private let processingQueue = DispatchQueue(label: "com.opencvdemo.preprocessing", qos: .background, attributes: .concurrent, autoreleaseFrequency: DispatchQueue.AutoreleaseFrequency.workItem, target: nil)
    private let detectAndComputeGroup = DispatchGroup()
    private let openCV = OpenCVWrapper()
    private let imagePicker = UIImagePickerController()
    private var statusLabel = UILabel()
    
    var image = UIImage()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        imageView.image = image
        imagePicker.delegate = self
        
        areaLabel.isHidden = true
        executionTimeLabel.isHidden = true
        
        statusLabel = UILabel(frame: CGRect(x: 0, y: imageView.frame.minY / 2, width: view.frame.width, height: imageView.frame.minY))
        statusLabel.textAlignment = .center
        statusLabel.font = UIFont(name: "System", size: 17.0)
        statusLabel.numberOfLines = 0
        statusLabel.text = "Press 'Found it!' and take a picture"
        
        view.addSubview(statusLabel)
    }
    
    func beginPreprocessingForQuery() {
        var isKeyline = false
        if keylinesToggle.isOn { isKeyline = true }
        processingQueue.async(group: detectAndComputeGroup) {
            var thumbnail = self.image
            if thumbnail.size.width > 1024 || thumbnail.size.height > 1024 {
                thumbnail = thumbnail.scaledTo(size: CGSize(width: 1024, height: 1024))!
            }
            
            if (isKeyline) {
                let _ = self.openCV.detectAndComputeWithKeylines(forQuery: thumbnail)
            } else {
                let _ = self.openCV.detectAndComputeWithKeypoints(forQuery: thumbnail)
            }
            print("Finished processing query")
        }
    }
    
    @IBAction func keylinesToggled(_ sender: Any) {
        beginPreprocessingForQuery()
    }
    
    @objc func tryAgain() {
        reactionButton.isHidden = true
        reactionButton.isUserInteractionEnabled = false
        
        imageView.image = image
        
        executionTimeLabel.isHidden = true
        areaLabel.isHidden = true
        
        statusLabel.text = "Press 'Found it!' and take a picture"
        
        foundItButton.isHidden = false
        foundItButton.isUserInteractionEnabled = true
        
        openCV.error = nil
        
        reactionButton.removeTarget(self, action: #selector(tryAgain), for: .touchUpInside)
        
        keylinesToggle.isHidden = false
        keylinesToggle.isUserInteractionEnabled = true
        
        keylinesToggleLabel.isHidden = false
    }
    
    @objc func findAnother() {
        openCV.reset()
        
        navigationController?.popViewController(animated: true)
    }
    
    @IBAction func foundIt(_ sender: Any) {
        if !keylinesToggle.isOn { beginPreprocessingForQuery() }
        
        imagePicker.sourceType = .photoLibrary
        
        present(imagePicker, animated: true)
    }
    
    // MARK: -UIImagePickerControllerDelegate
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [String : Any]) {
        var isKeyline = false
        if keylinesToggle.isOn { isKeyline = true }
        if let pickedImage = info[UIImagePickerControllerOriginalImage] as? UIImage {
            processingQueue.async(group: detectAndComputeGroup) {
                var thumbnail = pickedImage
                if thumbnail.size.width > 1024 || thumbnail.size.height > 1024 {
                    thumbnail = thumbnail.scaledTo(size: CGSize(width: 1024, height: 1024))!
                }
                
                if (isKeyline) {
                    let _ = self.openCV.detectAndComputeWithKeylines(forTrain: thumbnail)
                } else {
                    let _ = self.openCV.detectAndComputeWithKeypoints(forTrain: thumbnail)
                }
                
                print("Finished processing train")
            }
        } else {
            dismiss(animated: true, completion: nil)
        }

        dismiss(animated: true) {
            self.imageView.image = nil
            
            self.foundItButton.isHidden = true
            self.foundItButton.isUserInteractionEnabled = false
            
            self.keylinesToggle.isHidden = true
            self.keylinesToggle.isUserInteractionEnabled = false
            
            self.keylinesToggleLabel.isHidden = true
            
            self.statusLabel.text = "Processing..."
            
            let spinner = UIActivityIndicatorView()
            spinner.frame = CGRect(x: 0, y: 0, width: 40, height: 40)
            spinner.center = self.view.center
            spinner.hidesWhenStopped = true
            spinner.activityIndicatorViewStyle = .gray
            
            self.view.addSubview(spinner)
            spinner.startAnimating()
            
            self.detectAndComputeGroup.notify(queue: self.processingQueue) {
                var processedImage = UIImage()
                let startTime = Date()
                if isKeyline {
                    processedImage = self.openCV.matchWithKeylines()
                } else {
                    processedImage = self.openCV.matchWithKeypoints()
                }
                let endTime = Date()
                
                let executionTime = endTime.timeIntervalSince(startTime)
                
                print("Execution time: \(executionTime)")
                
                DispatchQueue.main.sync {
                    spinner.stopAnimating()
                    
                    if self.openCV.error != nil {
                        self.statusLabel.text = "Error finding match: \(self.openCV.error!)\n\nTry matching the example image exactly"
                        
                        self.reactionButton.setTitle("Try again?", for: .normal)
                        self.reactionButton.addTarget(self, action: #selector(self.tryAgain), for: .touchUpInside)

                    } else {
                        self.statusLabel.text = "You got it!"
                        
                        self.reactionButton.setTitle("Find another", for: .normal)
                        self.reactionButton.addTarget(self, action: #selector(self.findAnother), for: .touchUpInside)
                    }
                    
                    self.imageView.image = processedImage
                    
                    self.areaLabel.text = "Area: \(self.openCV.area)"
                    self.areaLabel.isHidden = false
                    
                    self.executionTimeLabel.text = "Execution time: \(executionTime)"
                    self.executionTimeLabel.isHidden = false
                    
                    self.reactionButton.isHidden = false
                    self.reactionButton.isUserInteractionEnabled = true
                }
            }
        }
    }

    func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
        dismiss(animated: true, completion: nil)
    }
}

extension UIImage {
    func scaledTo(size newSize: CGSize) -> UIImage? {
        let scale = min(size.width / newSize.width, size.height / newSize.height)

        let newSize = CGSize(width: size.width / scale, height: size.height / scale)
        let newOrigin = CGPoint(x: (newSize.width - newSize.width) / 2, y: (newSize.height - newSize.height) / 2)

        let smallRect = CGRect(origin: newOrigin, size: newSize).integral

        UIGraphicsBeginImageContextWithOptions(newSize, false, 0)

        draw(in: smallRect)

        let result = UIGraphicsGetImageFromCurrentImageContext()

        UIGraphicsEndImageContext()

        return result
    }
}

